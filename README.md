Spatial-Temporal LSTM network proposed in **Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network for Location Prediction[C]//IJCAI. 2018: 2341-2347.** Implemented with PyTorch.

- Core implementation is in `stlstm.py` - `STLSTMCell`.
- An example is presented in `stlstm_nextloc.py`. Some implementation is modified to fit into my task, but the calculation of slots and process of linear interpolation is general.
