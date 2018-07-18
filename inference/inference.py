import tensorflow as tf
import numpy as np
from tqdm import tqdm
import json
import pickle
import re
import nltk
import os
import sys
import math
from sklearn.metrics import accuracy_score

class DR_BILSTM():
	def __init__(self,embed_matrix1,num_rnn_layers,num_units,num_classes,lr,device):
		with tf.device(device):
			self.str1=tf.placeholder(tf.int32,shape=[None,None])
			self.sq1=tf.placeholder(tf.int32,shape=[None])
			self.sq2=tf.placeholder(tf.int32,shape=[None])
			self.str2=tf.placeholder(tf.int32,shape=[None,None])
			self.keep_prob=tf.placeholder(tf.float32)
			#this model is made as describe in the paper https://arxiv.org/abs/1802.05577


			with tf.name_scope('embed'):
				W = tf.get_variable(name = 'W', shape = embed_matrix1.shape, initializer = tf.constant_initializer(embed_matrix1), trainable = True)
				embeddings_out1 = tf.nn.embedding_lookup(W,self.str1)
				embeddings_out2=tf.nn.embedding_lookup(W,self.str2)
			#1st layer of bi-lstms
			with tf.variable_scope('bilstm11'):		
				rnn_layersfw1 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				rnn_layersbw1 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				multi_rnn_cellfw1 = tf.nn.rnn_cell.MultiRNNCell(rnn_layersfw1)
				multi_rnn_cellbw1=tf.nn.rnn_cell.MultiRNNCell(rnn_layersbw1)
				outputs1, state1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_rnn_cellfw1,cell_bw=multi_rnn_cellbw1,inputs=embeddings_out2,sequence_length=self.sq2,dtype=tf.float32)

			with tf.variable_scope('bilstm12'):
				rnn_layersfw2 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				rnn_layersbw2 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				multi_rnn_cellfw2 = tf.nn.rnn_cell.MultiRNNCell(rnn_layersfw2)
				multi_rnn_cellbw2=tf.nn.rnn_cell.MultiRNNCell(rnn_layersbw2)
				outputs2,state2 = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_rnn_cellfw2,cell_bw=multi_rnn_cellbw2,inputs=embeddings_out1,sequence_length=self.sq1,initial_state_fw=state1[0],initial_state_bw=state1[1])
				outputs2=tf.concat((outputs2[0],outputs2[1]),2)
			with tf.variable_scope('bilstm13'):
				rnn_layersfw3 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				rnn_layersbw3 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				multi_rnn_cellfw3 = tf.nn.rnn_cell.MultiRNNCell(rnn_layersfw3)
				multi_rnn_cellbw3=tf.nn.rnn_cell.MultiRNNCell(rnn_layersbw3)
				outputs3, state3 = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_rnn_cellfw3,cell_bw=multi_rnn_cellbw3,inputs=embeddings_out1,sequence_length=self.sq1,dtype=tf.float32)
			with tf.variable_scope('bilstm14'):
				rnn_layersfw4 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				rnn_layersbw4 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				multi_rnn_cellfw4 = tf.nn.rnn_cell.MultiRNNCell(rnn_layersfw4)
				multi_rnn_cellbw4=tf.nn.rnn_cell.MultiRNNCell(rnn_layersbw4)
				outputs4, state4 = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_rnn_cellfw4,cell_bw=multi_rnn_cellbw4,inputs=embeddings_out2,sequence_length=self.sq2,initial_state_fw=state3[0],initial_state_bw=state3[1])
				outputs4=tf.concat((outputs4[0],outputs4[1]),2)
			
			#attention layer according to paper
			def attention_u(ip):
				eij=tf.matmul(ip[0],tf.transpose(ip[1]))
				eij=tf.nn.softmax(eij,1)
				attention=tf.matmul(eij,ip[1])
				return attention,attention
			def attention_v(ip):
				eij=tf.matmul(ip[0],tf.transpose(ip[1]))
				eij=tf.nn.softmax(eij,1)
				attention=tf.matmul(eij,ip[1])
				return attention,attention
			u_attention=tf.map_fn(attention_u,(outputs2,outputs4))
			u_attention=u_attention[0]
			v_attention=tf.map_fn(attention_v,(outputs4,outputs2))
			v_attention=v_attention[0]

			#combining bilstm output and attention output according to paper
			uminusv=tf.subtract(outputs2,u_attention)
			uintov=tf.multiply(outputs2,u_attention)
			finalu=tf.concat((outputs2,u_attention,uminusv,uintov),2)
			vminusu=tf.subtract(outputs4,v_attention)
			vintou=tf.multiply(outputs4,v_attention)
			finalv=tf.concat((outputs4,v_attention,vminusu,vintou),2)

			#projecting the outputs to lower dimensions
			def project_v(ip):
				projected=tf.layers.dense(ip,num_units,activation=tf.nn.relu)
				return projected
			projected_v=tf.map_fn(project_v,finalv)
			def project_u(ip):
				projected=tf.layers.dense(ip,num_units,activation=tf.nn.relu)
				return projected
			projected_u=tf.map_fn(project_u,finalu)


			#2nd layer of bilstms
			with tf.variable_scope('bilstm21'):
				rnn_layersfw5 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				rnn_layersbw5 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				multi_rnn_cellfw5 = tf.nn.rnn_cell.MultiRNNCell(rnn_layersfw5)
				multi_rnn_cellbw5=tf.nn.rnn_cell.MultiRNNCell(rnn_layersbw5)
				outputs5, state5 = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_rnn_cellfw5,cell_bw=multi_rnn_cellbw5,inputs=projected_v,sequence_length=self.sq2,dtype=tf.float32)
				outputs5=tf.concat((outputs5[0],outputs5[1]),2)

			with tf.variable_scope('bilstm22'):
				rnn_layersfw6 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				rnn_layersbw6 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				multi_rnn_cellfw6 = tf.nn.rnn_cell.MultiRNNCell(rnn_layersfw6)
				multi_rnn_cellbw6=tf.nn.rnn_cell.MultiRNNCell(rnn_layersbw6)
				outputs6, state6 = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_rnn_cellfw6,cell_bw=multi_rnn_cellbw6,inputs=projected_u,sequence_length=self.sq1,initial_state_fw=state5[0],initial_state_bw=state5[1])
				outputs6=tf.concat((outputs6[0],outputs6[1]),2)
			with tf.variable_scope('bilstm23'):
				rnn_layersfw7 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				rnn_layersbw7 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				multi_rnn_cellfw7 = tf.nn.rnn_cell.MultiRNNCell(rnn_layersfw7)
				multi_rnn_cellbw7=tf.nn.rnn_cell.MultiRNNCell(rnn_layersbw7)
				outputs7, state7 = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_rnn_cellfw7,cell_bw=multi_rnn_cellbw7,inputs=projected_u,sequence_length=self.sq1,dtype=tf.float32)
				outputs7=tf.concat((outputs7[0],outputs7[1]),2)

			with tf.variable_scope('bilstm24'):
				rnn_layersfw8 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				rnn_layersbw8 = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size,activation=tf.nn.relu),output_keep_prob=self.keep_prob) for size in [num_units]*num_rnn_layers]
				multi_rnn_cellfw8 = tf.nn.rnn_cell.MultiRNNCell(rnn_layersfw8)
				multi_rnn_cellbw8=tf.nn.rnn_cell.MultiRNNCell(rnn_layersbw8)
				outputs8, state8 = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_rnn_cellfw8,cell_bw=multi_rnn_cellbw8,inputs=projected_v,sequence_length=self.sq2,initial_state_fw=state7[0],initial_state_bw=state7[1])	
				outputs8=tf.concat((outputs8[0],outputs8[1]),2)
			
			#maxpooling
			def maxpool(ip):
				ip1=tf.expand_dims(ip[0],axis=2)
				ip2=tf.expand_dims(ip[1],axis=2)
				ip_concat=tf.concat((ip1,ip2),axis=2)
				op=tf.reduce_max(ip_concat,2)
				return op,op
			outsentv=tf.map_fn(maxpool,(outputs5,outputs8))
			outsentu=tf.map_fn(maxpool,(outputs6,outputs7))
			outsentu=outsentu[0]
			outsentv=outsentv[0]
			
			#max and average pooling
			outsentumaxpool=tf.reduce_max(outsentu,1)
			outsentuavgpool=tf.reduce_mean(outsentu,1)
			outsentvmaxpool=tf.reduce_max(outsentv,1)
			outsentvavgpool=tf.reduce_mean(outsentv,1)
			final_concat_u_v=tf.concat((outsentumaxpool,outsentuavgpool,outsentvmaxpool,outsentvavgpool),1)
			

			#Final fully connected layers for classification
			fc1=tf.layers.dense(final_concat_u_v,num_units/2,activation=tf.nn.relu)
			self.fc2=tf.layers.dense(fc1,num_classes)
			self.labels=tf.placeholder(tf.int32,shape=[None])
			self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.fc2,labels=self.labels)
			
			self.loss=tf.reduce_mean(self.loss)
			self.opt=tf.train.AdamOptimizer(lr).minimize(self.loss)
			self.saver=tf.train.Saver()
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
def random_mini_batches(x1_train,x2_train,y_train,n,max1,max2,dictionary,mini_batch_size = 128):
	m=n
	mini_batches = []
	num_complete_minibatches = math.floor(m/mini_batch_size)
	for k in range(0, num_complete_minibatches):

		mini_batch_X1= x1_train[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch_X2=x2_train[k*mini_batch_size:k*mini_batch_size+mini_batch_size]
		mini_batch_Y = y_train[k * mini_batch_size  : k * mini_batch_size + mini_batch_size]
		seq1=[]
		seq2=[]

		arr1=np.zeros([mini_batch_size,max1])
		arr2=np.zeros([mini_batch_size,max2])
		c1=0
		for i in mini_batch_X1:
			sent=i.split()
			seq1.append(len(sent))
			c2=0
			for q in sent:
				arr1[c1,c2]=dictionary[q]
				c2+=1
			c1+=1
		c3=0

		for i in mini_batch_X2:
			sent=i.split()
			seq2.append(len(sent))
			c4=0
			for q in sent:
				arr2[c3,c4]=dictionary[q]
				c4+=1
			c3+=1
		

		mini_batch_seq1 = np.array(seq1)
		mini_batch_seq2=np.array(seq2)
		mini_batch_Y=np.array(mini_batch_Y,dtype=np.int32)
		mini_batch = (arr1,arr2, mini_batch_Y,mini_batch_seq1,mini_batch_seq2)
		mini_batches.append(mini_batch)
	
	return mini_batches

def run(data1,data2,max1,max2,labels,num_classes,numepochs,save_model_after_n_epochs,num_layers,lr,num_units,model_save_dir,dictionary,device,glove_vector_location,vocab_size,acc_file,loss_file,keep_prob=0.5,minibatch_size=32,testduringtrain=True,restore=False,trainingsetsize=204800):	
	words_to_index, index_to_words, word_to_vec_map = read_glove_vecs(glove_vector_location)
	embed_matrix1 = np.zeros((vocab_size, 300))#contains glove vectors for each word in vocabulary
	for word,index in dictionary.items():
		try:
			embed_matrix1[index, :] = word_to_vec_map[word]
		except:
			embed_matrix1[index, :] = np.random.uniform(-1, 1, 300)#if a word does not have a representation in the glove vectors we will randomly initialize it 
	meta_file_path='inference/meta.json'
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
	model=DR_BILSTM(embed_matrix1,num_layers,num_units,num_classes,lr,device)
	if trainingsetsize>len(data1):
		print('training set size is more than available')
		sys.exit()
	sentence1,test_sentence1=data1[:trainingsetsize],data1[trainingsetsize:trainingsetsize]
	sentence2,test_sentence2=data2[:trainingsetsize],data2[trainingsetsize:]
	label,test_label=labels[:trainingsetsize],labels[trainingsetsize:]
	config = tf.ConfigProto(allow_soft_placement = True)
	with tf.Session(config=config) as sess:
		if restore==True:
			model.saver.restore(sess,latest_model_path)
			print('model '+str(latest_model_path)+' restored')

		else:
			sess.run(tf.global_variables_initializer())
			print('new model started')
		accuracy={}
		lossgraph1={}
		for k in range(start_epoch_num+1,numepochs+1):
			minibatches=random_mini_batches(sentence1,sentence2,label,len(label),max1,max2,dictionary,minibatch_size)
			l=0
			preds=[]
			train_label=[]
			for m in tqdm(minibatches):
				(s1,s2,lb,seq1,seq2)=m
				out,ls,_=sess.run([model.fc2,model.loss,model.opt],feed_dict={model.str1:s1,model.str2:s2,model.sq1:seq1,model.sq2:seq2,model.labels:lb,model.keep_prob:keep_prob})
				l+=ls
				for p in range(out.shape[0]):
					h=np.argmax(out[p,:])
					preds.append(h)
					train_label.append(lb[p])
			print(str(k)+':loss:'+str(l/len(minibatches)))
			print(str(k)+':train_accuracy:'+str(accuracy_score(train_label,preds)))
			
			lossgraph1[k]=l/len(minibatches)
			if testduringtrain==True:
				minibatches_test=random_mini_batches(test_sentence1,test_sentence2,test_label,len(test_label),max1,max2,dictionary)
				test_preds=[]

				for m_t in minibatches_test:
					(s_t1,s_t2,lbt,seqt1,seqt2)=m_t
					ot=sess.run(model.fc2,feed_dict={model.str1:s_t1,model.str2:s_t2,model.sq1:seqt1,model.sq2:seqt2,model.labels:lbt,model.keep_prob:1.0})
					for i in range(ot.shape[0]):
						q=np.argmax(ot[i,:])
						test_preds.append(q)
				print(str(k)+':test_accuracy:'+accuracy_score(test_label,test_preds))
				accuracy[k]=[accuracy_score(train_label,preds),accuracy_score(test_label,test_preds)]
			else:
				accuracy[k]=[accuracy_score(train_label,preds)]


			with open(acc_file,'wb') as f:
				pickle.dump(accuracy,f)
			with open(loss_file,'wb') as f:
				pickle.dump(lossgraph1,f)
			if k%save_model_after_n_epochs==0:
				try:
					os.mkdir(model_save_dir+'/inference_model_'+str(k)+'/inf_model')
				except:
					pass
				model.saver.save(sess,model_save_dir+'/inference_model_'+str(k)+'/inf_model')
				json_data['model_path']=model_save_dir+'/inference_model_'+str(k)+'/inf_model'
				json_data['num_epochs_over']=k
				with open(meta_file_path,'w') as f:
					json.dump(json_data,f)