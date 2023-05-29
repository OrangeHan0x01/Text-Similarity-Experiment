import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
from transformers import BertModel, BertTokenizer
#from sklearn.model_selection import KFold
import json
import time
from visdom import Visdom

'''
#模型、参数、数据集声明节
'''
bert = BertModel.from_pretrained('../chinese_roBERTa_wwm_ext')
tokenizer = BertTokenizer.from_pretrained('../chinese_roBERTa_wwm_ext')

lr_bert = 1.5e-5
lr_fc = 2e-3
batch_size = 16
num_epochs = 2
max_len=168


f=open('../lcqmc/train.tsv','r',encoding='utf-8')
train_data=f.readlines()
f.close()
f_dev=open('../lcqmc/dev.tsv','r',encoding='utf-8')
test_data=f_dev.readlines()
f_dev.close()

'''
#数据集处理与加载节
'''
print('Loading train dataset...')

qp=[]
all_labels=[]
flag_1=0
for data_pre in train_data:
    data=data_pre.strip().split('\t')
    if(int(data[2])==0):
        qp.append(tokenizer.encode(data[0],data[1],add_special_tokens=True,max_length = max_len,padding='max_length'))
        all_labels.append(int(data[2]))
    elif(int(data[2])==1):
        qp.append(tokenizer.encode(data[0],data[1],add_special_tokens=True,max_length = max_len,padding='max_length'))
        all_labels.append(int(data[2]))
print('Loading test dataset...')
#print('num label_0:',d0,'num label_1:',d1)#num label_0: 100192 num label_1: 138574

qp_test=[]
all_labels_test=[]
for data_pre in test_data:
    data=data_pre.strip().split('\t')
    qp_test.append(tokenizer.encode(data[0],data[1],add_special_tokens=True,max_length = max_len,padding='max_length'))
    all_labels_test.append(int(data[2]))


qp = torch.tensor(qp)
qp_test = torch.tensor(qp_test)

labels=torch.tensor(all_labels)
labels_test=torch.tensor(all_labels_test)

dataset = TensorDataset(qp, labels)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataset_test = TensorDataset(qp_test, labels_test)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
train_flag=1

'''
#模型相关定义与声明节
'''

class BERTClassifier(nn.Module):#模型1：1层
  def __init__(self, bert):
    super().__init__()
    self.bert = bert
    layer_index=0
    for param in self.bert.parameters():#198、197--pooler层，181-196--11encoder.
        param.requires_grad = True
    for param in self.bert.embeddings.parameters():
        param.requires_grad = False
    self.fc1 = nn.Linear(bert.config.hidden_size, 1)#bert.config.hidden_size=768
    self.dropout=nn.Dropout(p=0.1)
  def forward(self, input_ids):
    pooled_output_1 = self.bert(input_ids)
    pooled_output_1 = pooled_output_1.pooler_output
    if(train_flag==1):
        pooled_output_1=self.dropout(pooled_output_1)
    logits = self.fc1(pooled_output_1)
    probs = torch.sigmoid(logits)
    return probs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model = BERTClassifier(bert)
model =model.to(device)
model.train()

bert_params = list(model.bert.parameters())
bert_optim = torch.optim.AdamW(bert_params, lr=lr_bert,weight_decay=3e-5)
fc1_params = list(model.fc1.parameters())
fc1_optim = torch.optim.AdamW(fc1_params, lr=lr_fc,weight_decay=3e-5)
loss_fn = nn.BCELoss()#sigmoid用
'''
#训练可视化声明节
'''
viz = Visdom()
viz.line([[0.,0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc']))
'''
#训练节
'''
total_batch=0
print('start train...')
running_loss=0
running_acc=0
for epoch in range(num_epochs):
  print('[+]new epoch!')
  for qp, labels in train_loader:
    total_batch+=1
    labels=labels.to(device)
    input_ids = qp.to(device)
    #model.zero_grad()#由于采用了多优化器，所以不再适用
    bert_optim.zero_grad()
    fc1_optim.zero_grad()
    logits = model(input_ids)
    logits = logits.view(-1)
    logits=logits.to(torch.float32)
    labels= labels.to(torch.float32)
    loss=loss_fn(logits, labels)
    correct = (logits > 0.5) == labels
    accuracy = correct.sum() / correct.numel()
    running_loss+=loss.item()
    running_acc+=accuracy.item()
    loss.backward()
    bert_optim.step()
    fc1_optim.step()
    if(total_batch%25==0):
      print('let the GPU take a break...')
      time.sleep(16)
      viz.line([[float(running_loss/25), float(running_acc/25)]], [total_batch/5], win='train', update='append')
      running_loss=0
      running_acc=0
  #每个epoch结束，对测试集进行检验
  print('total_batch: ',total_batch)
  test_batch=0
  test_acc_total=0
  train_flag=0
  for qp, labels in test_loader:
    test_batch+=1
    labels=labels.to(device)
    input_ids = qp.to(device)
    logits = model(input_ids).view(-1)
    logits=logits.to(torch.float32)
    labels= labels.to(torch.float32)
    correct = (logits > 0.5) == labels
    accuracy = correct.sum() / correct.numel()
    test_acc_total+=accuracy.item()
  print('test_acc: ',test_acc_total/test_batch)
  train_flag=1
  
#python -m visdom.server 启动可视化训练服务器
'''
#保存节
'''
#第3个epoch，每隔一定batch计算一次。超过之前的才保存。
save_index=0

test_batch=0
test_acc_total=0
train_flag=0
test_acc=0
print('start saving...')
for qp, labels in test_loader:
  test_batch+=1
  labels=labels.to(device)
  input_ids = qp.to(device)
  logits = model(input_ids).view(-1)
  logits=logits.to(torch.float32)
  labels= labels.to(torch.float32)
  correct = (logits > 0.5) == labels
  accuracy = correct.sum() / correct.numel()
  test_acc_total+=accuracy.item()
test_acc=int((1000*test_acc_total)/test_batch)
torch.save(model.bert.state_dict(), 'bert_model_'+str(save_index)+'_'+str(test_acc)+'.bin')
torch.save(model.fc1.state_dict(), 'fc_model_'+str(save_index)+'_'+str(test_acc)+'.bin')
save_index+=1
max_acc=test_acc

for qp, labels in train_loader:
  total_batch+=1
  labels=labels.to(device)
  input_ids = qp.to(device)
  bert_optim.zero_grad()
  fc1_optim.zero_grad()
  logits = model(input_ids).view(-1)
  logits=logits.to(torch.float32)
  labels= labels.to(torch.float32)
  loss=loss_fn(logits, labels)
  correct = (logits > 0.5) == labels
  accuracy = correct.sum() / correct.numel()
  running_loss+=loss.item()
  running_acc+=accuracy.item()
  loss.backward()
  bert_optim.step()
  fc1_optim.step()
  if(total_batch%25==0):
    print('let the GPU take a break...')
    time.sleep(16)
  if(total_batch%100==0):
    viz.line([[float(running_loss/100), float(running_acc/100)]], [total_batch/5], win='train', update='append')
    test_batch=0
    test_acc_total=0
    train_flag=0
    running_loss=0
    running_acc=0
    for qp, labels in test_loader:
      test_batch+=1
      labels=labels.to(device)
      input_ids = qp.to(device)
      logits = model(input_ids).view(-1)
      logits=logits.to(torch.float32)
      labels= labels.to(torch.float32)
      correct = (logits > 0.5) == labels
      accuracy = correct.sum() / correct.numel()
      test_acc_total+=accuracy.item()
    test_acc=test_acc_total/test_batch
    if(test_acc>=max_acc):
      torch.save(model.bert.state_dict(), 'bert_model_'+save_index+'_'+str(int(1000*test_acc))+'.bin')
      torch.save(model.fc1.state_dict(), 'fc_model_'+save_index+'_'+str(int(1000*test_acc))+'.bin')
      max_acc=test_acc
      save_index+=1
    train_flag=1
    print('let the GPU take a break...')
    time.sleep(20)
'''不想让bert参数变更时：
self.bert.requires_grad = False
optimizer = torch.optim.AdamW(self.fc.parameters(), lr=lr)
'''





'''
单个元素想要输入模型时：
d1=c1.view(1,-1)#c1=tensor([ 872, 1962, 8024, 2769, 1373, 3330, 1290])
bert(d1)
计算余弦相似度应该是用pooleroutput.view(-1).detach().numpy()

全程：（变成A和B的形式：）
q4="杯弓蛇影，张冠李戴"
c4=tokenizer.encode(q4,add_special_tokens=False)
d4=torch.tensor(c4).view(1,-1)
p4=bert(d4).pooler_output
pp4=p4.view(-1).detach().numpy()


q5="杯弓蛇影，张冠李戴"
c5=tokenizer.encode(q5,add_special_tokens=False,max_length = 20,padding='max_length')
d5=torch.tensor(c5).view(1,-1)
p5=bert(d5).pooler_output
pp5=p5.view(-1).detach().numpy()



GPU上的tensor不能和numpy直接转换。必须先转换为CPU上的tensor



import numpy as np
from numpy.linalg import norm

A = np.array([2,1,2])
B = np.array([3,4,2])

cosine = np.dot(A,B)/(norm(A)*norm(B))
cosine = np.dot(pp4,pp5)/(norm(pp4)*norm(pp5))
print("余弦相似度:", cosine)
得到的结果通常都在0.9以上。但确实有明显的区分相似度。
但是明显长度会造成很大影响，

'''