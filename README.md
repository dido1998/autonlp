# autonlp
# Project Title
This is a repository devoted to natural language processing and various applications machine learning in NLP.
Currently there is support for natural language generation and natural language inference.I hope to add more applications soon.

### Requirements
```
Tensorflow
Numpy
nltk
tqdm
```
Pretrained glove embeddings are required -
```
wget http://nlp.stanford.edu/data/glove.6B.zip 
unzip glove.6B.zip -d content
```
### Installing
```
git clone https://github.com/dido1998/autonlp
```
### Examples
```
python3 main.py --task <task> --datafile <your_data_file> --num_epochs <num_epochs> --model_save_path <path> --device <device name>
```
