import tensorflow as tf
from pathlib import Path
import sys
import numpy as np
import os
import json
from tqdm import tqdm
class basic_rnn_model():
	def __init__(self,num_layers,lr,rnn_block,embed_matrix1,end_token,num_units,fc1,fc2,vocab_size,device,max_seq_len_at_inf):
		
		self.inputs=tf.placeholder(tf.int32,shape=[None,None],name='inputs')#tensor of shape [batch_size,sequence_len] 
		self.sq=tf.placeholder(tf.int32,shape=[None],name='sequence_length')#sequence leangth for each instance
		with tf.name_scope('embed'):
			with tf.device(device):
				self.W = tf.get_variable(name = 'W', shape = embed_matrix1.shape, initializer = tf.constant_initializer(embed_matrix1), trainable = True)#embed_matrix1 is the pretrained glove embeddings
				self.embeddings_out = tf.nn.embedding_lookup(self.W, self.inputs)

		
		self.keep_prob=tf.placeholder(tf.float32,name='keep_probabaility')#dropout probobality
		self.labels=tf.placeholder(tf.int32,shape=[None,None])#tensor of shape [batch_size,sequence_length]
		self.starttoken=tf.placeholder(tf.int32,shape=[None])#index of the start token in the vocabulary 
		self.endtoken=tf.constant(end_token)#index of the end token in the vocabulary
		self.weights=tf.placeholder(tf.float32,shape=[None,None])

		
		self.train_helper = tf.contrib.seq2seq.TrainingHelper(self.embeddings_out, self.sq)

		self.pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.W, start_tokens=self.starttoken, end_token=self.endtoken)

		def decode(helper, scope,batch_size,reuse=None):
			with tf.variable_scope(scope, reuse=reuse):
				cell=None
				if rnn_block is 'LSTM':
					cell =[tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell (size,activation=tf.nn.relu),state_keep_prob=self.keep_prob) for size in [num_units]*num_layers]  
				else:
					cell =[tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.GRUCell (size,activation=tf.nn.relu),state_keep_prob=self.keep_prob) for size in [num_units]*num_layers]
				cell = tf.nn.rnn_cell.MultiRNNCell(cell, state_is_tuple=True)
				out_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.OutputProjectionWrapper(
				        cell,fc1,activation=tf.nn.relu,reuse=reuse),output_keep_prob=self.keep_prob)
				out_cell =  tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.OutputProjectionWrapper(
				        out_cell,fc2,activation=tf.nn.relu,reuse=reuse),output_keep_prob=self.keep_prob)
				out_cell = tf.contrib.rnn.OutputProjectionWrapper(out_cell,vocab_size,activation=None,reuse=reuse)
				decoder = tf.contrib.seq2seq.BasicDecoder(
				    cell=out_cell, helper=helper,
				    initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size))
				outputs = tf.contrib.seq2seq.dynamic_decode(
				    decoder=decoder,impute_finished=True, output_time_major=False,maximum_iterations=max_seq_len_at_inf
				     )
				return outputs[0]
		self.train_outputs = decode(self.train_helper, 'decode',tf.shape(self.inputs)[0])
		self.pred_outputs = decode(self.pred_helper, 'decode',tf.shape(self.starttoken)[0],reuse=True)
		self.train_outputs=self.train_outputs[0]# train_outputs is tensor of shape [batch_size,sequence_length,vocab_size]

		self.pred_outputs=self.pred_outputs[0]#pred_outputs is tensor of shape [batch_size,sequence_length,vocab_size]
		self.loss=tf.contrib.seq2seq.sequence_loss(logits=self.train_outputs,targets=self.labels,weights=self.weights)
		
		self.optimize=tf.train.AdamOptimizer(lr)
		self.grads=self.optimize.compute_gradients(self.loss)
		self.opt=self.optimize.apply_gradients(self.grads)

		self.saver=tf.train.Saver()
		self.graph=tf.get_default_graph()

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def run(data,numepochs,save_model_after_n_epochs,num_layers,lr,rnn_block,num_units,fc1,fc2,vocab_size,model_save_dir,dictionary,num_training_examples,device,max_seq_len_at_inf,glove_vector_location,keep_prob=0.5,minibatch_size=32,testduringtrain=True,restore=False):
	
	words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(glove_vector_location)
	embed_matrix1 = np.zeros((vocab_size, 300))#contains glove vectors for each word in vocabulary
	for word,index in dictionary.items():
		try:
			embed_matrix1[index, :] = word_to_vec_map[word]
		except:
			embed_matrix1[index, :] = np.random.uniform(-1, 1, 300)#if a word does not have a representation in the glove vectors we will randomly initialize it 
	endtoken=dictionary['<end>']
	meta_file_path='generation/meta.json'
	meta_file=None
	json_data={}
	if os.path.exists(meta_file_path):
		if restore==False:
			meta_file=open(meta_file_path,'w')
		else:
			meta_file=open(meta_file_path,'r')
			json_data=json.load(meta_file)
			meta_file.close()
	else:
		if restore==True:
			print('no saved model found')
			print('exiting...')
			sys.exit()

	
	latest_model_path=None
	start_epoch_num=0
	if restore==True:
		start_epoch_num=json_data['num_epochs_over']
		latest_model_path=json_data['model_path']
	meta_file.close()
	model=basic_rnn_model(num_layers,lr,rnn_block,embed_matrix1,endtoken,num_units,fc1,fc2,vocab_size,device,max_seq_len_at_inf)
	
	with tf.Session(graph=model.graph) as sess:
		if restore==True:
			model.saver.restore(sess,latest_model_path)
			print('model '+str(latest_model_path)+' restored')

		else:
			sess.run(tf.global_variables_initializer())
			print('new model started')
		print('entering training loop')
		for k in range(start_epoch_num+1,numepochs+1):
			l=0
			for i in tqdm(range(int(num_training_examples/minibatch_size))):
				curr_batch=data[i*minibatch_size:i*minibatch_size+minibatch_size]
				max_seq_len_in_curr_batch=0
				for c in curr_batch:
					h11=c.split()
					if len(h11)>max_seq_len_in_curr_batch:
						max_seq_len_in_curr_batch=len(h11)
				lbs=np.zeros([minibatch_size,max_seq_len_in_curr_batch+1])#array to store labels
				inps=np.zeros([minibatch_size,max_seq_len_in_curr_batch+1])#array to store inputs
				ws=np.zeros([minibatch_size,max_seq_len_in_curr_batch+1])#array to store weights
				#ws keeps a track of the length of each sequence.This is passed to tf.contrib.seq2seq.sequence_loss() in the weights argument.
				for x in range(minibatch_size):
					inps[x,0]=dictionary['<start>']
					ws[x,0]=1
				c1=0
				sqlen=np.zeros([minibatch_size])
				for c in curr_batch:
					h=c.split()
					c2=1
					c3=0
					sqlen[c1]=len(h)+1
					for j in h:
						ws[c1,c2]=1
				
						lbs[c1,c3]=dictionary[j]
						inps[c1,c2]=dictionary[j]
						c2+=1
						c3+=1
					c1+=1
				loss,_=sess.run([model.loss,model.opt],feed_dict={model.sq:sqlen,model.keep_prob:keep_prob,model.weights:ws,model.starttoken:np.array([dictionary['<start>']]),model.inputs:inps,model.labels:lbs})
				l+=loss

			if testduringtrain==True:
				f=sess.run(model.pred_outputs,feed_dict={model.starttoken:np.array([dictionary['<start>']]),model.keep_prob:1.0})
				a1=f[0,:,:]
				
				preds=''
				for i in range(a1.shape[0]):
					w1=np.argmax(a1[i,:])	
					for q,t in dictionary.items():
						if t==w1:
							preds=preds+" "+q
				print(preds)
			curr_batch=[]

			l=l/(num_training_examples/minibatch_size)

			print(str(k)+':loss:'+str(l))
			if k%save_model_after_n_epochs==0:
				try:

					os.mkdir(model_save_dir+'/generation_model_'+str(k)+'/gen_model')
				except:
					pass
				model.saver.save(sess,model_save_dir+'/generation_model_'+str(k)+'/gen_model')
				json_data['model_path']=model_save_dir+'/generation_model_'+str(k)+'/gen_model'
				json_data['num_epochs_over']=k
				with open(meta_file_path,'w') as f:
					json.dump(json_data,f)