import struct
import numpy
import base64
import json
import time

def vec2str(vec):
	struct_id=''
	for i in range(768):
		struct_id+='f'
	bt=struct.pack(struct_id,* vec)
	strb64=base64.b64encode(bt).decode('gbk')#len=4096
	return strb64

def str2vec(b64str):
	struct_id=''
	for i in range(768):
		struct_id+='f'
	bt2=base64.b64decode(b64str.encode('gbk'))
	newmt=struct.unpack(struct_id,bt2)
	return numpy.array(newmt,dtype='float32')

def dataset_transform(similarity):#将数据集转化为向量集存储，向量集格式：每行以逗号分割3个值：向量1，向量2，标签
	with open('../lcqmc/oppp.json','r') as f:
		data=json.load(f)
		train_data=data['train']
		test_data=data['dev']
	data_train=[]
	data_test=[]
	i=0
	for data in train_data:
		inputn=[[[data['q1'],data['q2']]]]
		output=similarity.task_instance._preprocess(inputs=inputn)
		vecs=similarity.task_instance._run_model_vec(inputs=output)['result']
		vec1,vec2=vec2str(vecs[0]),vec2str(vecs[1])
		data_train.append([vec1,vec2,data['label']])
		i+=1
		if(i%700==0):
			print('sleep..',i/700)
			time.sleep(10)
	with open('./train_vecs.json','w') as f_train:
		json.dump(data_train,f_train)
	print('train_data finished!')
	for data in test_data:
		input=[[[data['q1'],data['q2']]]]
		output=similarity.task_instance._preprocess(inputs=input)
		vecs=similarity.task_instance._run_model_vec(inputs=output)['result']
		vec1,vec2=vec2str(vecs[0]),vec2str(vecs[1])
		data_test.append([vec1,vec2,data['label']])
		i+=1
		if(i%700==0):
			print('sleep..',i/700)
			time.sleep(10)
	with open('./test_vecs.json','w') as f_test:
		json.dump(data_test,f_test)

		
def get_train_vecs():#将向量集取出为[[向量1，向量2，标签],...],167173条
	vec_array=[]
	with open('./train_vecs.json','r') as f:
		dataset=json.load(f)
	for pair in dataset:
		vec1=str2vec(pair[0])
		vec2=str2vec(pair[1])
		label=pair[2]
		vec_array.append([vec1,vec2,label])
	return vec_array

def get_test_vecs():#将向量集取出为[[向量1，向量2，标签],...]，10000条
	vec_array=[]
	with open('./test_vecs.json','r') as f:
		dataset=json.load(f)
	for pair in dataset:
		vec1=str2vec(pair[0])
		vec2=str2vec(pair[1])
		label=pair[2]
		vec_array.append([vec1,vec2,label])
	return vec_array

def create_simset(vec_array):#从向量集创建相似度数据集
	sim_set=[]
	for pair in vec_array:
		similarity = (pair[0] * pair[1]).sum(axis=0)
		sim_set.append([similarity,pair[2]])
	return sim_set






def dataset_transform_tsv(similarity):
	with open('../lcqmc/dev.tsv','r',encoding='utf-8') as f:
		test_data=f.readlines()
	data_test=[]
	i=0
	for data_pre in test_data:
		data=data_pre.strip().split('\t')
		inputn=[[[data[0],data[1]]]]
		output=similarity.task_instance._preprocess(inputs=inputn)
		vecs=similarity.task_instance._run_model_vec(inputs=output)['result']
		vec1,vec2=vec2str(vecs[0]),vec2str(vecs[1])
		data_test.append([vec1,vec2,data[2]])
		i+=1
		if(i%600==0):
			print('sleep..',i/600)
			time.sleep(10)
	with open('./lcdev_vecs.json','w') as f_dev:
		json.dump(data_test,f_dev)

def get_lcdev_vecs():#将向量集取出为[[向量1，向量2，标签],...]，10000条
	vec_array=[]
	with open('./lcdev_vecs.json','r') as f:
		dataset=json.load(f)
	for pair in dataset:
		vec1=str2vec(pair[0])
		vec2=str2vec(pair[1])
		label=pair[2]
		vec_array.append([vec1,vec2,label])
	return vec_array

def dataset_transform_traintsv(similarity):
	with open('../lcqmc/train.tsv','r',encoding='utf-8') as f:
		test_data=f.readlines()
	data_test=[]
	i=0
	for data_pre in test_data:
		data=data_pre.strip().split('\t')
		inputn=[[[data[0],data[1]]]]
		output=similarity.task_instance._preprocess(inputs=inputn)
		vecs=similarity.task_instance._run_model_vec(inputs=output)['result']
		vec1,vec2=vec2str(vecs[0]),vec2str(vecs[1])
		data_test.append([vec1,vec2,data[2]])
		i+=1
		if(i%600==0):
			print('sleep..',i/600)
			time.sleep(10)
	with open('./lct_vecs.json','w') as f_dev:
		json.dump(data_test,f_dev)

def get_lctrain_vecs():#将向量集取出为[[向量1，向量2，标签],...]，10000条
	vec_array=[]
	with open('./lct_vecs.json','r') as f:
		dataset=json.load(f)
	for pair in dataset:
		vec1=str2vec(pair[0])
		vec2=str2vec(pair[1])
		label=pair[2]
		vec_array.append([vec1,vec2,label])
	return vec_array

