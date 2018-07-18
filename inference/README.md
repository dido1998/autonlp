### NATURAL LANGUAGE INFERENCE
This model is based on the paper https://arxiv.org/abs/1802.05577.

The current code is based on the SNLI dataset.For any other dataset the main.py file can be changed as required. 
## Example
```
python3 main.py --task inference --datafile <your data file> --num_epochs <num_epochs> --model_save_path <path> --device <device> --glove_vector_location <location> --inf_num_rnn_layers <num_layers> --inf_num_rnn_units <num_units> --inf_minibatch_size <size> --acc_file <file> --loss_file <file> --num_classes <number of classes>
```
