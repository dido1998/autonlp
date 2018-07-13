from tensorflow import flags
import nltk
import re
import numpy as np
import generation.tensorflow_generate as runner
import os, shutil

Flags=flags.FLAGS


class Generation():
	def clean_str(self,string):
		string = string.strip().lower()

		string = re.sub(r"\s{2,}", " ", string)
		string = re.sub(r',', '', string)
		string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
		if 'let\'s' in string:
		    string = re.sub(r'let\'s', 'let us', string)
		if 'lets' in string:
		    string = re.sub(r'lets', 'let us', string)

		string = re.sub(r"\'s", " is", string)
		string = re.sub(r"\'ve", " have", string)
		if 'wont ' in string:
		    string = re.sub(r"won\'?t", "will not", string)
		if 'won\'t ' in string:
		    string = re.sub(r"won\'?t", "will not", string)

		if 'cant ' in string:
		    string = re.sub(r"n\'?t", " can not", string)
		if 'can\'t ' in string:
		    string = re.sub(r"n\'?t", " can not", string)

		string = re.sub(r"n\'t", " not", string)
		string = re.sub(r"\'re", " are", string)
		string = re.sub(r"\'d", " \'d", string)
		string = re.sub(r"\'ll", " will", string)
		string = re.sub(r",", " , ", string)
		string = re.sub(r"!", "", string)
		string = re.sub(r"\(", " \( ", string)
		string = re.sub(r"\)", " \) ", string)
		string = re.sub(r"\?", "", string)
		string = re.sub(r"\s{2,}", " ", string)
		# string = re.sub(r"\'", '', string)

		return string.strip()
	def preprocess(self,sentence):
		sentence = sentence.lower()
		sentence=self.clean_str(sentence)

		tokenizer = nltk.RegexpTokenizer(r'\w+')

		tokens = tokenizer.tokenize(sentence)
		m=len(tokens)
		tokens=" ".join(tokens)
		return m,tokens
	def get_data(self):
		#this funtion assumes that data is present in a text file with each instance on seperate line
		#this function can be altered as required to get data from different files.
		#it returns the sentences as a list and the dictionary
		f=open(Flags.datafile,'r')
		text=f.read();

		sentence=text.split('\n')
		processed_sent=[]


		max=0
		for s in sentence:
			m,sent=self.preprocess(s)
			if m>max:
				max=m
			processed_sent.append(sent)
		dictionary={}
		dictionary['<start>']=0
		dictionary['<end>']=1
		for p in processed_sent:
			words=p.split()
			for w in words:
				if w not in dictionary:
					dictionary[w]=len(dictionary)
		return processed_sent,dictionary
	def run(self):
		data,dictionary=self.get_data()
		print('got data from '+str(Flags.datafile))

		try:
			os.mkdir(Flags.model_save_path)
			print('making directory '+str(Flags.model_save_path)+' to save models')
		except:
			print('saving model to '+str(Flags.model_save_path)+' directory')
			if Flags.restore_model==False:
				folder = Flags.model_save_path
				for the_file in os.listdir(folder):
					file_path = os.path.join(folder, the_file)
					try:
						if os.path.isfile(file_path):
						    os.unlink(file_path)
						elif os.path.isdir(file_path): shutil.rmtree(file_path)
					except Exception as e:
						pass
		 
		runner.run(data,Flags.num_epochs,Flags.save_model_after_n_epochs,Flags.num_rnn_layers,Flags.learning_rate,Flags.rnn_block,Flags.num_units,Flags.fc1,Flags.fc2,len(dictionary),Flags.model_save_path,dictionary,len(data),Flags.device,Flags.max_seq_len_at_inference,Flags.glove_vector_location,testduringtrain=Flags.testduringtrain,keep_prob=Flags.keep_prob,restore=Flags.restore_model,minibatch_size=Flags.minibatch_size)



if __name__=='__main__':
	flags.DEFINE_string("datafile",'/home/aniket/nlp/bbc/image_coco.txt','file which has the data')
	flags.DEFINE_string("task",'generation','currently only language generation is supported')
	flags.DEFINE_integer("num_epochs",10,'number of epochs')
	flags.DEFINE_integer('save_model_after_n_epochs',1,'number of epochs after which to save model')
	#generation specific parameters
	flags.DEFINE_string("rnn_block",'LSTM','can be GRU or LSTM')
	flags.DEFINE_integer('num_units',32,'number of units in each lstm layer')
	flags.DEFINE_string("architecture of the block",'','can be uni-directional or bi-directional')
	flags.DEFINE_integer("num_rnn_layers",2,"number of layers in the recurrent neural network")
	flags.DEFINE_integer("fc1",128,'num units in first fc layer')
	flags.DEFINE_integer("fc2",256,'num units in second fc layer')
	flags.DEFINE_integer('minibatch_size',32,'batch size to use')
	flags.DEFINE_float('learning_rate',5e-4,'Learning rate for generation')
	flags.DEFINE_float('keep_prob',0.5,'dropout keep probability')
	flags.DEFINE_string('model_save_path','generation/model','directory to save model')
	flags.DEFINE_bool('testduringtrain',True,'enable testing during training')
	flags.DEFINE_bool('restore_model',False,'True for restore model , False for starting new model')
	flags.DEFINE_string('device','/device:CPU:0','device to train the model on')
	flags.DEFINE_integer('max_seq_len_at_inference',45,'set this to maximum length of an instance in your training set')
	flags.DEFINE_string('glove_vector_location','/home/aniket/nlp/bbc/glove.txt','location of the pretrained glove vectors')
	###glove vectors can be obtained by executing the following commands
	###wget http://nlp.stanford.edu/data/glove.6B.zip 
	###unzip glove.6B.zip -d content
	g=Generation()
	g.run()