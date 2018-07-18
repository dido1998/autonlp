## Text generation

The model consists of multilayered lstm cells with fully connected layers on the top.

## Example
```
python3 main.py --task generation --datafile <datafile> --num_epochs <num_epochs> --rnn_block LSTM --num_units <num_units> --num_rnn_layers <num layers>
```
This command has to be executed from the directory containing main.py

## Results
I trained the lstm model using the following parameters-\n
num_layers:4\n
num_features(rnn):512\n
fc1:1024\n
fc2:2048\n
learning_rate:5e-3\n
batch_size:100\n

The model was trained on a dataset collected from image captions.The dataset can found here-
```
https://github.com/geek-ai/Texygen/blob/master/data/image_coco.txt
```
Some of the generated sentences-
```
a person presses her hand to the window of a large oven.
a woman sitting on top of a kitchen counter.
```

