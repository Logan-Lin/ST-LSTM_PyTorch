import json
import math
import os
import pickle
from bisect import bisect

import nni
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from torch import nn
from tqdm import tqdm, trange

from module.stlstm import STLSTM
from nextloc import get_embed
from utils import next_batch

nni_training = True
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def cal_slot_distance(value, slots):
    """
    Calculate a value's distance with nearest lower bound and higher bound in slots.

    :param value: The value to be calculated.
    :param slots: values of slots, needed to be sorted.
    :return: normalized distance with lower bound and higher bound,
        and index of lower bound and higher bound.
    """
    higher_bound = bisect(slots, value)
    lower_bound = higher_bound - 1
    if higher_bound == len(slots):
        return 1., 0., lower_bound, lower_bound
    else:
        lower_value = slots[lower_bound]
        higher_value = slots[higher_bound]
        total_distance = higher_value - lower_value
        return (value - lower_value) / total_distance, \
               (higher_value - value) / total_distance, \
               lower_bound, higher_bound


def cal_slot_distance_batch(batch_value, slots):
    """
    Proceed `cal_slot_distance` on a batch of data.

    :param batch_value: a batch of value, size (batch_size, step)
    :param slots: values of slots, needed to be sorted.
    :return: batch of distances and indexes. All with shape (batch_size, step).
    """
    # Lower bound distance, higher bound distance, lower bound, higher bound.
    ld, hd, l, h = [], [], [], []
    for batch in batch_value:
        ld_row, hd_row, l_row, h_row = [], [], [], []
        for step in batch:
            ld_one, hd_one, l_one, h_one = cal_slot_distance(step, slots)
            ld_row.append(ld_one)
            hd_row.append(hd_one)
            l_row.append(l_one)
            h_row.append(h_one)
        ld.append(ld_row)
        hd.append(hd_row)
        l.append(l_row)
        h.append(h_row)
    return np.array(ld), np.array(hd), np.array(l), np.array(h)


def construct_slots(min_value, max_value, num_slots, type):
    """
    Construct values of slots given min value and max value.

    :param min_value: minimum value.
    :param max_value: maximum value.
    :param num_slots: number of slots to construct.
    :param type: type of slots to construct, 'linear' or 'exp'.
    :return: values of slots.
    """
    if type == 'exp':
        n = (max_value - min_value) / (math.exp(num_slots - 1) - 1)
        return [n * (math.exp(x) - 1) + min_value for x in range(num_slots)]
    elif type == 'linear':
        n = (max_value - min_value) / (num_slots - 1)
        return [n * x + min_value for x in range(num_slots)]


class Dataset:
    """
    Dataset class for training ST-LSTM classifier.
    """
    def __init__(self, train_data, val_data, history_count,
                 poi_count):
        """
        :param train_data: pandas DataFrame containing the training dataset.
        :param val_data: pandas DataFrame containing the validation dataset.
        :param history_count: length of historical sequence in every training set.
        :param poi_count: total count of POIs.
        """
        self.history_count = history_count
        self.poi_count = poi_count

        self.min_t, self.max_t, self.min_d, self.max_d = 1e8, 0., 1e8, 0.

        self.train_pair = self.construct_sequence(train_data)
        self.val_pair = self.construct_sequence(val_data)
        self.train_size = len(self.train_pair)
        self.val_size = len(self.val_pair)

        _, _, _, self.val_label = zip(*self.val_pair)

    def construct_sequence(self, data):
        """
        Construct history sequence and label pairs for training.

        :param data: pandas DataFrame containing the dataset.
        :return: pairs of history sequence and label.
        """
        # Preprocess dataset, calculate time delta and distances
        # between sequential visiting records.
        data_ = pd.DataFrame(data, copy=True)
        data_.index -= 1
        data_.columns = [f'{c}_' for c in data.columns]
        data = pd.concat([data, data_], axis=1).iloc[1:-1]
        data['delta_t'] = (data['time_'] - data['time']).apply(lambda time: time.seconds)
        data['delta_d'] = ((data['latitude'] - data['latitude_']).pow(2) +
                           (data['longitude'] - data['longitude_']).pow(2)).pow(0.5)
        data['user_id_'] = data['user_id_'].astype(int)
        data['poi_id_'] = data['poi_id_'].astype(int)
        data['user_id'] = data['user_id'].astype(int)
        data['poi_id'] = data['poi_id'].astype(int)
        data = data[data['user_id'] == data['user_id_']]

        # Update the min and max value of time delta and distance.
        self.min_t = min(self.min_t, data['delta_t'].min())
        self.max_t = max(self.max_t, data['delta_t'].max())
        self.min_d = min(self.min_d, data['delta_d'].min())
        self.max_d = max(self.max_d, data['delta_d'].max())

        # Construct history and label pairs.
        pairs = []
        for user_id, group in tqdm(data.groupby('user_id'),
                                   total=data['user_id'].drop_duplicates().shape[0],
                                   desc='Construct sequences'):
            if group.shape[0] > self.history_count:
                for i in range(group.shape[0] - self.history_count):
                    his_rows = group.iloc[i:i+self.history_count]
                    history_location = his_rows['poi_id_'].tolist()
                    history_t = his_rows['delta_t'].tolist()
                    history_d = his_rows['delta_d'].tolist()
                    label_location = group.iloc[i+self.history_count]['poi_id_']
                    pairs.append((history_location, history_t, history_d, label_location))
        return pairs

    def train_iter(self, batch_size):
        return next_batch(shuffle(self.train_pair), batch_size)

    def val_iter(self, batch_size):
        return next_batch(self.val_pair, batch_size)


class STLSTMClassifier(nn.Module):
    """
    RNN classifier using ST-LSTM as its core.
    """
    def __init__(self, input_size, output_size, hidden_size,
                 temporal_slots, spatial_slots,
                 device, learning_rate):
        """
        :param input_size: The number of expected features in the input vectors.
        :param output_size: The number of classes in the classifier outputs.
        :param hidden_size: The number of features in the hidden state.
        :param temporal_slots: values of temporal slots.
        :param spatial_slots: values of spatial slots.
        :param device: The name of the device used for training.
        :param learning_rate: Learning rate of training.
        """
        super(STLSTMClassifier, self).__init__()
        self.temporal_slots = sorted(temporal_slots)
        self.spatial_slots = sorted(spatial_slots)

        # Initialization of network parameters.
        self.st_lstm = STLSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        # Embedding matrix for every temporal and spatial slots.
        self.embed_s = nn.Embedding(len(temporal_slots), input_size)
        self.embed_s.weight.data.normal_(0, 0.1)
        self.embed_q = nn.Embedding(len(spatial_slots), input_size)
        self.embed_q.weight.data.normal_(0, 0.1)

        # Initialization of network components.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

        self.device = torch.device(device)
        self.to(self.device)

    def place_parameters(self, ld, hd, l, h):
        ld = torch.from_numpy(np.array(ld)).type(torch.FloatTensor).to(self.device)
        hd = torch.from_numpy(np.array(hd)).type(torch.FloatTensor).to(self.device)
        l = torch.from_numpy(np.array(l)).type(torch.LongTensor).to(self.device)
        h = torch.from_numpy(np.array(h)).type(torch.LongTensor).to(self.device)

        return ld, hd, l, h

    def cal_inter(self, ld, hd, l, h, embed):
        """
        Calculate a linear interpolation.

        :param ld: Distances to lower bound, shape (batch_size, step)
        :param hd: Distances to higher bound, shape (batch_size, step)
        :param l: Lower bound indexes, shape (batch_size, step)
        :param h: Higher bound indexes, shape (batch_size, step)
        """
        # Fetch the embed of higher and lower bound.
        # Each result shape (batch_size, step, input_size)
        l_embed = embed(l)
        h_embed = embed(h)
        return torch.stack([hd], -1) * l_embed + torch.stack([ld], -1) * h_embed

    def forward(self, batch_l, batch_t, batch_d):
        """
        Process forward propagation of ST-LSTM classifier.

        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :return: prediction result of this batch, size (batch_size, output_size, step).
        """
        batch_l = torch.from_numpy(np.array(batch_l)).type(torch.FloatTensor).to(self.device)

        t_ld, t_hd, t_l, t_h = self.place_parameters(*cal_slot_distance_batch(batch_t, self.temporal_slots))
        d_ld, d_hd, d_l, d_h = self.place_parameters(*cal_slot_distance_batch(batch_d, self.spatial_slots))

        batch_s = self.cal_inter(t_ld, t_hd, t_l, t_h, self.embed_s)
        batch_q = self.cal_inter(d_ld, d_hd, d_l, d_h, self.embed_q)

        hidden_out, cell_out = self.st_lstm(batch_l, batch_s, batch_q)
        linear_out = self.linear(hidden_out[:,-1,:])
        return linear_out

    def predict(self, batch_l, batch_t, batch_d):
        """
        Predict a batch of data.

        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :return: batch of predicted class indices, size (batch_size).
        """
        return torch.max(self.forward(batch_l, batch_t, batch_d), 1)[1].detach().cpu().numpy().squeeze()


def batch_train(model: STLSTMClassifier, batch_l, batch_t, batch_d, batch_label):
    """
    Train model using one batch of data and return loss value.

    :param model: One instance of STLSTMClassifier.
    :param batch_l: batch of input location sequences,
        size (batch_size, time_step, input_size)
    :param batch_t: batch of temporal interval value, size (batch_size, step)
    :param batch_d: batch of spatial distance value, size (batch_size, step)
    :param batch_label: batch of label, size (batch_size)
    :return: loss value.
    """
    prediction = model(batch_l, batch_t, batch_d)
    batch_label = torch.from_numpy(np.array(batch_label)).type(torch.LongTensor).to(model.device)

    model.optimizer.zero_grad()
    loss = model.loss_func(prediction, batch_label)
    loss.backward()
    model.optimizer.step()

    return loss.detach().cpu().numpy()


def predict(model: STLSTMClassifier, batch_l, batch_t, batch_d):
    """
    Predict a batch of data using ST-LSTM classifier.

    :param model: One instance of STLSTMClassifier.
    :param batch_l: batch of input location sequences,
        size (batch_size, time_step, input_size)
    :param batch_t: batch of temporal interval value, size (batch_size, step)
    :param batch_d: batch of spatial distance value, size (batch_size, step)
    :return: batch of predicted class indices, size (batch_size).
    """
    return torch.max(model(batch_l, batch_t, batch_d), 1)[1].detach().cpu().numpy().squeeze()


def get_dataset(dataset_name, history_count, test, force_build=False):
    """
    Get certain dataset for training.
    Will read from cache file if cache exists, or build from scratch if not.

    :param dataset_name: name of dataset.
    :param history_count: length of historical sequence in training set.
    :param test: Use test set or validation set to build this dataset.
    :param force_build: Ignore the existence of cache file and re-build dataset.
    :return: a instance of Dataset class.
    """
    cache_basedir = os.path.join('data', 'stlstm', 'cache')
    if not os.path.exists(cache_basedir):
        os.makedirs(cache_basedir)

    cache_filepath = os.path.join(cache_basedir,
                                  f'{dataset_name}_h{history_count}_t{test}.dataset')
    if os.path.exists(cache_filepath) and not force_build:
        with open(cache_filepath, 'rb') as cache_fp:
            dataset = pickle.load(cache_fp)
    else:
        hdf_filedir = os.path.join('data', 'data-split', f'{dataset_name}.h5')
        train_df = pd.read_hdf(hdf_filedir, key='train')
        val_df = pd.read_hdf(hdf_filedir, key='test' if test else 'val')

        with open(os.path.join('data', 'data-split', f'{dataset_name}.json'), 'r') as meta_fp:
            meta_data = json.load(meta_fp)
        dataset = Dataset(train_df, val_df, history_count=history_count,
                          poi_count=meta_data['poi'])
        with open(cache_filepath, 'wb') as cache_fp:
            pickle.dump(dataset, cache_fp)

    return dataset


def train(dataset, embed_matrix, display_batch, hidden_size,
          device, training_epochs, batch_size, learning_rate,
          num_temporal_slots, num_spatial_slots,
          temporal_slot_type, spatial_slot_type):
    """
    Train a ST-LSTM model using given dataset and embedding vectors.

    :param dataset: a instance of Dataset class, containing training dataset.
    :param embed_matrix: A matrix $Z\in \mathbb R^{N\times D}$,
        with $i$-th row being the embedding vector of $i$-th location.
    :param display_batch: Number of batches to train before run
        through the whole validation set to get a new set of metric value.
    :param hidden_size: The number of features in the hidden state.
    :param device: name of device to train this model.
    :param training_epochs: Total number of epochs to train.
    :param batch_size: batch size.
    :param learning_rate: learning rate.
    :param num_temporal_slots: number of temporal slots to construct.
    :param num_spatial_slots: number of spatial slots to construct.
    :param temporal_slot_type: type of temporal slots to construct.
    :param spatial_slot_type: type of spatial slots to construct.
    :return: the trained prediction model and accuracy log on validation set.
    """
    temporal_slots = construct_slots(dataset.min_t, dataset.max_t,
                                     num_temporal_slots, temporal_slot_type)
    spatial_slots = construct_slots(dataset.min_d, dataset.max_d,
                                    num_spatial_slots, spatial_slot_type)

    model = STLSTMClassifier(input_size=embed_matrix.shape[1], output_size=dataset.poi_count,
                             hidden_size=hidden_size, device=device,
                             learning_rate=learning_rate,
                             temporal_slots=temporal_slots, spatial_slots=spatial_slots)

    acc_list = []
    trained_batches = 0
    with trange(training_epochs * math.ceil(dataset.train_size / batch_size), desc='Training') as bar:
        for epoch in range(training_epochs):
            for train_batch in dataset.train_iter(batch_size):
                batch_l, batch_t, batch_d, batch_label = zip(*train_batch)
                _ = batch_train(model, embed_matrix[np.array(batch_l)], batch_t, batch_d, batch_label)
                trained_batches += 1
                bar.update(1)

                if trained_batches % display_batch == 0:
                    bar.set_description('Testing')
                    pres = []
                    for test_batch in dataset.val_iter(batch_size):
                        test_l, test_t, test_d, test_label = zip(*test_batch)
                        pre_batch = predict(model, embed_matrix[np.array(test_l)], test_t, test_d).tolist()
                        if isinstance(pre_batch, int):
                            pres.append(pre_batch)
                        else:
                            pres += pre_batch
                    pres = np.array(pres)

                    f1_micro = f1_score(dataset.val_label, pres, average='micro')
                    if nni_training:
                        nni.report_intermediate_result(f1_micro)
                        acc_list.append(f1_micro)
                    else:
                        f1_macro = f1_score(dataset.val_label, pres, average='macro')
                        prec_micro = precision_score(dataset.val_label, pres, average='micro')
                        prec_macro = precision_score(dataset.val_label, pres, average='macro')
                        recall_micro = recall_score(dataset.val_label, pres, average='micro')
                        recall_macro = recall_score(dataset.val_label, pres, average='macro')

                        acc_list.append([f1_micro, f1_macro, prec_micro, prec_macro,
                                        recall_micro, recall_macro])

                    bar.set_description('f1_micro %.7f' % f1_micro)
    return model, np.array(acc_list)


if __name__ == '__main__':
    if nni_training:
        args = nni.get_next_parameter()
        dataset = get_dataset(dataset_name=args["dataset"], history_count=3, test=True)
        embed_filename = os.path.join('tale', 'embed',
                                      f'{args["dataset"]}_slice{args["slice"]}_span{args["influence"]}_'
                                      f'c2_size200_epoch8_lr0.0004_'
                                      f'batch16.embed.npy')
        embed_matrix = get_embed(embed_filename, poi_count=dataset.poi_count, embed_size=200)
        _, acc_log = train(dataset=dataset, embed_matrix=embed_matrix,
                           display_batch=2000, hidden_size=200, device='cuda:0',
                           training_epochs=40, batch_size=32, learning_rate=1e-4,
                           num_temporal_slots=args["numt"], num_spatial_slots=args["nums"],
                           temporal_slot_type=args["typet"], spatial_slot_type=args["types"])
        nni.report_final_result(np.max(acc_log))
    else:
        dataset = get_dataset(dataset_name='nyc', history_count=3, test=True)
        embed_matrix = get_embed(embed_file=os.path.join('tale', 'embed', 'nyc_slice120_span0_c2_size200_epoch16_lr0.0002_batch8.embed.npy'),
                                 poi_count=dataset.poi_count, embed_size=200)
        _, acc_log = train(dataset=dataset, embed_matrix=embed_matrix,
                           display_batch=1000, hidden_size=200, device='cpu',
                           training_epochs=40, batch_size=32, learning_rate=1e-4,
                           num_temporal_slots=10, num_spatial_slots=10,
                           temporal_slot_type='linear', spatial_slot_type='linear')
        print(np.max(acc_log, axis=0))
