#!/usr/bin/env python
# coding: utf-8

# In[1]:


#5个随机种子模型，不区分验证集 8706 3 vote score:8817
# 5个随机种子模型，不区分验证集 8306 4 vote  score: 0.88571744662


# In[2]:


# ! unzip ../data/results.zip -d ../data/
# ! unzip ../data/train.zip -d ../data/
# ! unzip ../data/test.zip -d ../data/
# ! unzip ../data/sample_example.zip -d ../data/


# In[3]:


train_data_path = '../data/sample_example/'
test_data_path = '../data/test/'


# # import

# In[4]:


import lr_scheduler as L
from optim import Optim
import pandas as pd
import collections
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from functools import reduce
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from bert4keras.tokenizers import Tokenizer
import os
from transformers import *
from transformers import BertModel
from torchcrf import CRF
from tqdm.notebook import tqdm as tqdm_notebook


# In[ ]:


def to_set(df):
    '''df转set'''
    df = df[['text_id',  'from_index', 'to_index', 'entity','word']].sort_values(['text_id','from_index'])
    s = []
    for i,line in df.iterrows():
        s.append(tuple(line))
    return set(s)
def readAnn(filename,text_id):
    '''读取的单个文件信息'''
    ann_info = pd.read_csv(filename,header=None,engine='python')
#     print(ann_info)
    ann_info = ann_info[0].str.split('#',expand=True)
    ann_info.columns = [ 'from_index', 'to_index','entity','word']
    ann_info['text_id'] = text_id
    ann_info = ann_info[['text_id', 'entity', 'from_index', 'to_index', 'word']]
    ann_info = ann_info.astype({'text_id':'int','entity':'str','from_index':'int','to_index':'int','word':'str'})
    return ann_info
def readResults(name):
    '''从文件夹读取标注信息，返回Dataframe'''
    res_path = '../submit/' + name
    ann = []
    for file_name in os.listdir(res_path):
        if file_name=='.ipynb_checkpoints':
            continue
        text_id = file_name.split('.')[0]
        file_name = res_path + '/'+ file_name
        ann.append(readAnn(file_name,text_id))
    #     break
    return pd.concat(ann)


# In[5]:


LABELS =['Tloc', 'This', 'Tdiff', 'Tnum', 'Tsize', 'MVI', 'TNM', 'LC',
       'Sate'] 

train_data_path = '../data/train/'
test_data_path = '../data/test/'

WIN_SIZE = 508  #切分文本的窗口大小
WIN_STEP = 300 #
MAX_LEN = WIN_SIZE+4

epoch = 10
batch_size = 4
SEED = 1210
SEP_TOKEN_ID = 102
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# bert配置
dict_path = './UER/bert_model/vocab.txt'
TOKENIZER = Tokenizer(dict_path, do_lower_case=True)
BERT_MODEL_PATH = './UER/bert_model/'
BERT_SIZE = 1024
#模型存储配置
LAST_MODEL_NUM = 2

MODEL_DIR = '../model/baseline' #不加/
if MODEL_DIR.split('/')[-1] not in os.listdir('../model/'):
    os.mkdir(MODEL_DIR)

#预测配置
VOTE_THRESHOLD = 2

#lr 参数
learning_rate = 5e-5
max_grad_norm = 10
learning_rate_decay = 0.99
start_decay_at = 6

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(SEED)


# In[6]:


def getLabels2Id():
    '''BIO标记法，获得tag:id 映射字典'''
    labels = ['O']
    for label in LABELS:
        labels.append('B-' + label)
        labels.append('I-' + label)
    labels2id = {label: id_ for id_, label in enumerate(labels)}
    id2labels = {id_: label for id_, label in enumerate(labels)}
    return labels2id, id2labels
LABELS2ID , ID2LABELS = getLabels2Id()
def save_variable(v,filename):
    with open(filename,'wb') as f:
        pickle.dump(v,f)
    return filename
 
def load_variavle(filename):
    with open(filename,'rb') as f:
        r=pickle.load(f)
    return r


# # 清洗训练集

# In[7]:


get_ipython().system(' mv ../submit/train_data_tag/results/*  ../data/train/txt/')


# In[17]:


train_data_path = '../data/train/'
test_data_path = '../data/test/'
#读取训练集
def readdata(text_id,file_name):
    with open(file_name  + '.txt',encoding='gbk') as f:
        content = f.readlines()
    content = ''.join([x.replace('\n','||') for x in content])
    ann_info = pd.read_csv(file_name + '.tag',header=None)
    ann_info = ann_info[0].str.split('#',expand=True)[[0,1,2,3]]
    ann_info.columns = [ 'from_index', 'to_index','entity','word']
    ann_info['text_id'] = text_id
    ann_info = ann_info[['text_id', 'entity', 'from_index', 'to_index', 'word']]
    ann_info = ann_info.astype({'text_id':'int','entity':'str','from_index':'int','to_index':'int','word':'str'})
    return content,ann_info
def writeTrainfile(i,text,ann_info):
    text_file = train_data_path + 'data/' + str(i) + '.txt'
    ann_file =  train_data_path + 'label/' + str(i) +'.csv' 
    with open(text_file,'w+',encoding='utf-8') as f:
        f.write(text)
    ann_info.to_csv(ann_file,index=None)
    
#读取100样例
buf = []
content,ann_info = '',[]
shift = 0
for i in tqdm_notebook([21,22]):
    f = '../data/sample_example/%s' % (i)
    con,ann = readdata(i,f)
    content += con
    ann_info.append(ann)
    ann['from_index'] += shift
    ann['to_index'] += shift
    shift += len(con)
for i in tqdm_notebook(list(range(20))):
    f = '../data/train/txt/%s' % (i)
    if not os.path.exists(f+'.tag'): #文件不存在 跳过
        continue
    con,ann = readdata(i,f)
    content += con
    ann_info.append(ann)
    ann['from_index'] += shift
    ann['to_index'] += shift
    shift += len(con)
    
ann_info = pd.concat(ann_info)   
texts = content.split('||||')
shift = 0
for i,text in enumerate(texts):
    sentence_len = len(text)
    a = ann_info[(ann_info.from_index>shift)&(ann_info.to_index<shift + sentence_len)]
    a['from_index'] -= shift
    a['to_index'] -= shift
    shift += sentence_len+4
    if sentence_len>1:
        buf.append((text,a))
#写成训练格式      
for i,(text,a) in enumerate(buf):
    writeTrainfile(i,text,a)


# In[18]:


max_i =  i
max_i


# # 训练集读取

# In[21]:


class Feature():
    '''特征基类'''
    def __init__(self, text_id, or_text, text, role_arguments,content_split_off,shift):
        self.text_id = text_id
        self.or_text = or_text
        self.text = text
        self.ann_info = role_arguments
        self.content_split_off = content_split_off
        self.shift = shift
        
    def __str__(self):
        return self.__repr__()
    def getFeature(self):
        pass
    def __repr__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
    
class BertFeature(Feature):
    '''Bert特征类'''
    def __init__(self, text_id, or_text, text, role_arguments,content_split_off,shift):
        super(BertFeature, self).__init__(text_id, or_text, text,role_arguments, content_split_off,shift)
        self.getFeature()  #提取特征
        self.check()

    def getFeature(self):
        tokens = TOKENIZER.tokenize(self.text)  #分词
        self.mapping = TOKENIZER.rematch(self.text, tokens)
        self.token_ids = TOKENIZER.tokens_to_ids(tokens)
        if len(self.token_ids) > MAX_LEN:
            raise ValueError('token_ids 大于 MAX_LEN!',self)
        if len(self.token_ids) < MAX_LEN:
                self.token_ids += [0] * (MAX_LEN - len(self.token_ids))  #padding
        self.segment_ids = [0] * len(self.token_ids)
        self.labels = np.zeros(len(self.token_ids)).astype(int) 
        for index in self.ann_info.index:
            ann_row = self.ann_info.loc[index]
            ann_len = len(TOKENIZER.tokens_to_ids(TOKENIZER.tokenize(ann_row.word))) - 2#对一个实体分词并转换为token_ids  实体分词后占的长度
            if int(ann_row.from_index) >= self.content_split_off and int(ann_row.from_index) <= MAX_LEN+self.content_split_off:#该实体标注在当前text中
                for i in range(1,len(self.mapping)-1):
                    if self.content_split_off + self.mapping[i][0] == int(ann_row.from_index):
                        self.labels[i] = LABELS2ID.get('B-'+ann_row.entity)
                        self.labels[i+1:i+ann_len] = LABELS2ID.get('I-'+ann_row.entity)
                        
    def check(self):
        if self.or_text[self.content_split_off] != self.text[0]:
            raise ValueError('content_split_off error!')
        if len(self.labels) != len(self.token_ids) or len(self.mapping)==len(self.token_ids):
            print(len(self.labels),len(self.token_ids),len(segment_ids))
            print(self.__repr__())
            raise ValueError('打标错误！')

class TrainSet():
    def __init__(self, data_path, mode='train'):
        '''初始化'''
        self.data_path = data_path #文件目录
        self.mode = mode 
        self.cnt = 0
        self.Features = []
        print('读取训练数据...')
        filenamepair = {
                data_path + 'data/' + str(i) + '.txt':
                data_path + 'label/' + str(i) + '.csv'
                for i in range(max_i)
            }
#         print(filenamepair)
        for i, (text_file_name, ann_file_name) in enumerate(filenamepair.items()):  #读取所有文本数据
            text = self.readText(text_file_name)
            if self.mode == 'train':  #提取标签
                ann_info = self.readAnnInfo(ann_file_name)
                ann_info['to_index'] -= 1 
            else:
                ann_info = pd.DataFrame()
            text_id_ = int(os.path.splitext(text_file_name)[0].split('/')[-1]) 
            self.Features += self.splitStr2(text_id_,0,text,ann_info)
            
        self.cnt = len(self.Features)            

    def splitStr2(self,text_id,shift,text,ann_info, win_size=WIN_SIZE,step=WIN_STEP):
        '''
        将一条训练数据用滑窗法切分为多条训练数据
        '''
        subtexts = []
        begain_index = 0
        while begain_index + win_size < len(text):
            buf = text[begain_index:begain_index + win_size]
            if buf[-1]!='||':
                buf = ''.join([x+'||' for x in buf.split('||')[:-1]])
#                 print(begain_index,buf)
                if ann_info.shape[0] != 0: #测试集的ann_info 为 0 
                    sub_ann_info = ann_info[(ann_info.from_index>=begain_index)&(ann_info.from_index<=begain_index+WIN_SIZE)]
                else:
                    sub_ann_info = ann_info
                subtexts.append(
                    BertFeature(text_id, text,buf,sub_ann_info, begain_index,shift))
            begain_index += step
        if begain_index<len(text):
#             print(begain_index,text[begain_index:begain_index + win_size])
            subtexts.append(
                BertFeature(text_id, text,text[begain_index:begain_index + win_size], ann_info,begain_index,shift))
        return subtexts            
    def __repr__(self):
        return self.mode + '\n' + '样本量:' + str(self.cnt) + '\n' + str(
            self.mode)
    def __len__(self):
        '''获取该数据类型的长度'''
        return len(self.Features)
    def __getitem__(self,index):
        '''根据index获取元素'''
        feature = self.Features[index]
        token_ids = feature.token_ids
        seg_ids = feature.segment_ids
        labels = np.array(feature.labels)
        return token_ids, seg_ids, labels

    def readText(self, filename):
        '''读取文本'''
        with open(filename, encoding='utf-8') as f:
            text = f.read()
        return text

    def readAnnInfo(self,filename):
        '''根据.ann文件名读取标注信息'''
        data = pd.read_csv(filename,encoding='utf-8')
        data.columns = ['text_id', 'entity', 'from_index', 'to_index', 'word']
        data['to_index'] = data['to_index'] + 1
        data['word'] = data['word'].astype(str)
        data = data[data.entity.isin(LABELS)]
        return data.drop_duplicates()
    

train_set_path = 'train.pkl'
test_set_path = 'test.pkl'

    
# if train_set_path in os.listdir('../data/'):
#     train_set = load_variavle('../data/'+train_set_path)
#     print('读取成功！')
# else :
#     train_set = TrainSet(train_data_path)
#     save_variable(train_set,'../data/'+train_set_path)
#     print('保存成功！')
train_set = TrainSet(train_data_path)
save_variable(train_set,'../data/'+train_set_path)
print('保存成功！')


# # 测试集读取

# In[22]:


class TestSet():
    def __init__(self, data_path, mode='test'):
        '''初始化'''
        self.data_path = data_path #文件目录
        self.mode = mode 
        self.cnt = 0
        self.Features = []
        print('读取训练数据...')
        filenamepair = {
                data_path + 'txt/' + str(i) + '.txt':
                data_path + 'label/' + str(i) + '.csv'
                for i in range(21)
            }
#         print(filenamepair)
        for i, (text_file_name, ann_file_name) in enumerate(filenamepair.items()):  #读取所有文本数据
            texts = self.readText(text_file_name).split('||||')
            print(text_file_name,'has',len(texts))
            shift = 0
            ann_info = pd.DataFrame()
            text_id_ = int(os.path.splitext(text_file_name)[0].split('/')[-1])
            for text in texts:
                sentence_len = len(text)
                if sentence_len >1: #句子长度大于1
                     self.Features += self.splitStr2(text_id_,shift,text,ann_info)
                shift += sentence_len + 4 
        self.cnt = len(self.Features)
        
    def splitStr2(self,text_id,shift,text,ann_info, win_size=WIN_SIZE,step=WIN_STEP):
        '''
        将一条训练数据用滑窗法切分为多条训练数据
        '''
        subtexts = []
        begain_index = 0
        while begain_index + win_size < len(text):
            buf = text[begain_index:begain_index + win_size]
            if buf[-1]!='||':
                buf = ''.join([x+'||' for x in buf.split('||')[:-1]])
#                 print(begain_index,buf)
                if ann_info.shape[0] != 0: #测试集的ann_info 为 0 
                    sub_ann_info = ann_info[(ann_info.from_index>=begain_index)&(ann_info.from_index<=begain_index+WIN_SIZE)]
                else:
                    sub_ann_info = ann_info
                subtexts.append(
                    BertFeature(text_id, text,buf,sub_ann_info, begain_index,shift))
            begain_index += step
        if begain_index<len(text):
#             print(begain_index,text[begain_index:begain_index + win_size])
            subtexts.append(
                BertFeature(text_id, text,text[begain_index:begain_index + win_size], ann_info,begain_index,shift))
        return subtexts    
    def __repr__(self):
        return self.mode + '\n' + '样本量:' + str(self.cnt) + '\n' + str(
            self.mode)
    def __len__(self):
        '''获取该数据类型的长度'''
        return len(self.Features)
    def __getitem__(self,index):
        '''根据index获取元素'''
        feature = self.Features[index]
        token_ids = feature.token_ids
        seg_ids = feature.segment_ids
        labels = np.array(feature.labels)
        return token_ids, seg_ids, labels

    def readText(self, filename):
        '''读取文本'''
        with open(filename, encoding='gbk') as f:
            content = f.readlines()
        return ''.join([x.replace('\n','||') for x in content])

# if test_set_path in os.listdir('../data/'):
#     test_set = load_variavle('../data/'+test_set_path)
#     print('读取成功！')
# else:
#     test_set = TestSet(test_data_path, mode='test')
#     save_variable(test_set,'../data/'+test_set_path)
#     print('保存成功！')
test_set = TestSet(test_data_path, mode='test')


# # dataloader

# In[23]:


def getInputFeature(batch):
#     print([len(x.token_ids) for x in batch])
    batch_token_ids = torch.stack([torch.tensor(x.token_ids) for x in batch])
    batch_segment_ids = torch.stack([torch.tensor(x.segment_ids) for x in batch])
    batch_labels = torch.stack([torch.tensor(x.labels) for x in batch])
    return batch_token_ids,batch_segment_ids,batch_labels
def getDataLoder(df,batch_size,train_mode = False):
    loader = torch.utils.data.DataLoader(df,batch_size=batch_size,shuffle=train_mode,collate_fn=getInputFeature)
    return loader,df


# # model
# 

# In[24]:


class MyBertModel(nn.Module):
    def __init__(self):
        num_labels=len(LABELS)*2+1
        super(MyBertModel, self).__init__() #初始化
        self.model_name = 'my_bert_model'
        self.bert_model = BertModel.from_pretrained(BERT_MODEL_PATH,cache_dir=None,output_hidden_states=True) #加载Bert模型，
#         self.bert_model2 = BertModel.from_pretrained(BERT_MODEL_PATH2,cache_dir=None,output_hidden_states=True) #加载Bert模型，
        self.zy_hidden_fc= nn.Sequential(nn.Linear(BERT_SIZE, num_labels))#
        self.crf=CRF(num_labels,batch_first=True) #不加batch_first 默认第二维是batch
    
    def mask_mean(self,x,mask):
        mask_x=x*(mask.unsqueeze(-1))
        x_sum=torch.sum(mask_x,dim=1)
        re_x=torch.div(x_sum,torch.sum(mask,dim=1).unsqueeze(-1))
        return re_x
    
    def forward(self,ids,seg_ids,labels,is_test=False):
        attention_mask = (ids > 0)
        last_seq,pooled_output,hidden_state=self.bert_model(input_ids=ids,token_type_ids=seg_ids,attention_mask=attention_mask)
#         last_seq2,pooled_output2,hidden_state2=self.bert_model2(input_ids=ids,token_type_ids=seg_ids,attention_mask=attention_mask)
#         bert_cat = torch.cat((last_seq,last_seq2),2)
        emissions=self.zy_hidden_fc(last_seq)
        if not is_test:
            loss=-self.crf(emissions, labels, mask=attention_mask,reduction='mean')
            return loss
        else:
            decode=self.crf.decode(emissions,attention_mask)
            return decode
def debug_label():
    dataloader,feature = getDataLoder(train_set.Features[:64],2,train_mode=True)
    model=MyBertModel()
    for token_ids, seg_ids,labels, in dataloader:
        print(token_ids.size())
        y = model(token_ids, seg_ids,labels,is_test=False)
        print(y)
        y = model(token_ids, seg_ids,labels,is_test=True)
        print(y)
        print(len(y))
        break
# debug_label()


# In[13]:


# train valid


# In[25]:


def getCateryMetric(ture_labels,pre_labels):
    pre_ture = ture_labels & pre_labels
    orther_f1 = {} 
    for tpe in LABELS:
        orther_f1[tpe+'_total'] = 0
        orther_f1[tpe+'_predict'] = 0
        orther_f1[tpe+'_true'] = 0
    for i in ture_labels:
        text_id,category, sta,end = i[0],i[3],i[1],i[2]
        orther_f1[category+'_total'] =  orther_f1[category+'_total']+1
    for i in pre_labels:
        text_id,category, sta,end =  i[0],i[3],i[1],i[2]
        orther_f1[category+'_predict'] =  orther_f1[category+'_predict']+1  
    for i in pre_ture:
        text_id,category, sta,end =  i[0],i[3],i[1],i[2]
        orther_f1[category+'_true'] =  orther_f1[category+'_true']+1
    cate_ = {}    
    for tpe in LABELS:
        if tpe+'_total' in orther_f1.keys():
            metric = {}
            total = orther_f1[tpe+'_total']
            pre = orther_f1[tpe+'_predict']
            true = orther_f1[tpe+'_true']
            p=true/pre if pre>0 else 0.
            r=true/total if total>0 else 0.
            f=(2*p*r)/(p+r) if (p+r)>0 else 0.
            metric['acc'] = p
            metric['recall'] = r
            metric['f1'] = f
            metric['total_number'] = total
            metric['predict_number'] = pre
            metric['predict_score'] = true
#             print(tpe,':',metric)
            cate_[tpe] = metric  
    return cate_
def metric_fn(predict_res,true_res,log=False):
    ##for log
    log_cnt=0
    X,Y,Z = 1e-10, 1e-10, 1e-10 #防止除零
#     print('--------predict------',predict_res)
#     print('--------true------',true_res)   
    
    for res in predict_res:
        if log and log_cnt < 0:
            print(res)
            log_cnt += 1
#     print('*'*20+'没有预测到的有:'+'*'*20+'\n',true_res-predict_res)
#     print('*'*20+'多余的预测结果:'+'*'*20+'\n',predict_res-true_res)
    #compute score
    # compute category score
    a = getCateryMetric(true_res,predict_res)
    for k in a:
        print(k,a[k])
    #
    X += len(predict_res & true_res)
    Y += len(predict_res)
    Z += len(true_res)
    print(X,Y,Z)
#     return predict_res,true_res
    return 2*X/(Y+Z),X/Y,X/Z
def validation_fn(model,val_loader,val_features,is_test=False,val=True,log=True):
    model.eval() #不启用 BatchNormalization 和 Dropout,否则会改变模型权值
    bar = tqdm_notebook(val_loader)
    decodes=[] #存放decode 结果
    for i,(token_ids, seg_ids,labels) in enumerate(bar):
        decode= model(token_ids.to(DEVICE),seg_ids.to(DEVICE),labels.to(DEVICE),is_test=True)
        decodes.extend(decode)
    #挨个预测
    
    t_ans = []
    p_ans = []
#     print(text)
    for index in tqdm_notebook(range(len(decodes))):#遍历每个答案
        decode=decodes[index]
        feat=val_features[index]#Feature class
        item_id=feat.text_id
        text=feat.or_text
        off=1 #cls
        mapping=feat.mapping #映射
        role_arguments=feat.ann_info
        content_split_off=feat.content_split_off
        start_flag = False
        buf = ''
        start_index = -1
        
        for loc in role_arguments.index:
            row = role_arguments.loc[loc]
            t_ans.append((item_id,int(row.from_index),int(row.to_index),row.entity,row.word))
#*********************************************************************************************************************
        starting=False
        predict_set=[]
        for i,label in enumerate(decode[1:-1]): #decode[1:end]
            i = i+1
            if label > 0:
                if label%2==1: #B
                    starting=True
                    predict_set.append(([i],LABELS[label//2])) #([i],label)·
                elif starting: #I
                    predict_set[-1][0].append(i) 
                else: #
                    starting=False
            else:
                starting=False
        for indexs,typ in predict_set: 
#             print(indexs,typ,mapping)
            fr = mapping[indexs[0]][0]
            to = mapping[indexs[-1]][-1]+1
            ans_start = int(fr+content_split_off)
            ans_end = int(to+content_split_off)
            an = text[ans_start:ans_end]
#             print((item_id,ans_start,ans_end,typ,an))
            p_ans.append((item_id,ans_start,ans_end,typ,an))
            

    if not is_test:
        re=metric_fn(set(p_ans),set(t_ans),log=False)
        return re
    else:
        return list(set(p_ans)),decodes
def train_model(model,train_loader,val_loader,val_features,val_loader_recall=None,val_features_recall=None,accumulation_steps=1,early_stop_epochs=2,epochs=4,model_save_path='pytorch_zy_model_true.pkl'):  
    
    losses=[]
    ########梯度累计
    batch_size = accumulation_steps*32
    ########早停
    no_improve_epochs=0
    
    ########优化器 学习率

    optimizer = Optim('adamw', 
                  learning_rate, 
                  max_grad_norm,
                  lr_decay=learning_rate_decay, 
                  start_decay_at=start_decay_at)
    optimizer.set_parameters(model.parameters())
    scheduler = L.CosineAnnealingLR(optimizer.optimizer, T_max=epochs)
    train_len=len(train_loader)
    best_vmetric=-np.inf
    tloss = []
    
    for epoch in range(1,epochs+1):
        model.train()
        bar = tqdm_notebook(train_loader)
        bar.set_description("traing epoch %s"%epoch)
        for i,(token_ids, seg_ids,labels) in enumerate(bar):
            loss= model(token_ids.to(DEVICE),seg_ids.to(DEVICE),labels.to(DEVICE),is_test=False)
            sloss=loss
            sloss.backward()
            tloss.append(loss.item())
            if (i+1) % accumulation_steps == 0 or (i+1)==train_len: #
                optimizer.step()
                scheduler.step()
                optimizer.optimizer.zero_grad()
            bar.set_postfix(loss=np.array(tloss).mean())
        
        #val
        val_f1,val_p,val_recall=validation_fn(model,val_loader,val_features)
        if val_features_recall:
            res_test=validation_fn(model,val_loader_recall,val_features_recall,log=False)
            print('test f1:',str(res_test))
            losses.append(str(res_test))
        losses.append( 'train_loss:%.5f, f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (np.array(tloss).mean(),val_f1, val_p, val_recall, best_vmetric))
        print('epoch:',epoch,losses[-1])
        
        if epoch >= epochs - LAST_MODEL_NUM: #存最后n 轮
            torch.save(model.state_dict(),model_save_path+'epoch_{}'.format(epoch))
#             model.load_state_dict(torch.load(model_save_path+'epoch_{}'.format(epoch)))
        if val_f1>=best_vmetric: #如果
            torch.save(model.state_dict(),model_save_path)
            best_vmetric=val_f1
            no_improve_epochs=0
            print('improve save model!!!')
        else:
            no_improve_epochs+=1
#         if no_improve_epochs==early_stop_epochs:
#             print('no improve score !!! stop train !!!') #early stop
#             break
    return losses


# In[ ]:





# # CV

# In[26]:


torch.cuda.empty_cache()
from sklearn.model_selection import KFold
import pickle
# valid_data = [x for x in train_set.Features if x.text_id<10000] #原始训练数据

FOLD=5
SEED = list(range(FOLD))
kf = KFold(n_splits=FOLD, shuffle=True,random_state=2020)
log_losses=[] 
train_preds_dict=collections.defaultdict(list) 
for i,(train_index , test_index) in enumerate(kf.split(train_set.Features)):
    seed_everything(SEED[i])
    print(str(i+1),'*'*50)
    tra=[train_set.Features[i] for i in train_index]
    valid=[train_set.Features[i] for i in test_index]
    print(len(tra))
    print(len(valid))
    valid_loader,valid_features=getDataLoder(valid,batch_size=batch_size,train_mode=False) 
    model_save_path=MODEL_DIR+'/ner_large{}'.format(i+1)
    model=MyBertModel().to(DEVICE)
    if i>=0:
        tra_loader,tra_features=getDataLoder(tra,batch_size=batch_size,train_mode=True)
        losses=train_model(model,tra_loader,valid_loader,valid_features,accumulation_steps=1,early_stop_epochs=20,epochs=epoch,model_save_path=model_save_path)
        log_losses.extend(losses)
    # 加载最好模型
    print(str(i+1),'-'*50)
    log_losses.append('*'*50)
    torch.cuda.empty_cache()


# # predict

# In[27]:


def predict(model,val_loader,val_features,is_test=False,val=True,log=True):
    model.eval() #不启用 BatchNormalization 和 Dropout,否则会改变模型权值
    bar = tqdm_notebook(val_loader)
    decodes=[] #存放decode 结果
    for i,(token_ids, seg_ids,labels) in enumerate(bar):
        decode= model(token_ids.to(DEVICE),seg_ids.to(DEVICE),labels.to(DEVICE),is_test=True)
        decodes.extend(decode)
    #挨个预测
    
    t_ans = []
    p_ans = []
#     print(text)
    for index in tqdm_notebook(range(len(decodes))):#遍历每个答案
        decode=decodes[index]
        feat=val_features[index]#Feature class
        item_id=feat.text_id
        text=feat.or_text
        shift = feat.shift
        off=1 #cls
        mapping=feat.mapping #映射
        role_arguments=feat.ann_info
        content_split_off=feat.content_split_off
        start_flag = False
        buf = ''
        start_index = -1
        
        for loc in role_arguments.index:
            row = role_arguments.loc[loc]
            t_ans.append((item_id,int(row.from_index),int(row.to_index),row.entity,row.word))
#*********************************************************************************************************************
        starting=False
        predict_set=[]
        for i,label in enumerate(decode[1:-1]): #decode[1:end]
            i = i+1
            if label > 0:
                if label%2==1: #B
                    starting=True
                    predict_set.append(([i],LABELS[label//2])) #([i],label)·
                elif starting: #I
                    predict_set[-1][0].append(i) 
                else: #
                    starting=False
            else:
                starting=False
        for indexs,typ in predict_set: 
#             print(indexs,typ,mapping)
            fr = mapping[indexs[0]][0]
            to = mapping[indexs[-1]][-1]+1
            ans_start = int(fr+content_split_off)
            ans_end = int(to+content_split_off)
            an = text[ans_start:ans_end]
            p_ans.append((item_id,ans_start+shift,ans_end+shift,typ,an))
    return list(set(p_ans)),decodes


# In[28]:


test_loader,test_features  = getDataLoder(test_set.Features,batch_size=8,train_mode=False)
test_preds=[] 
decodes = [] 
for model_path in tqdm_notebook(os.listdir(MODEL_DIR)):
    if 'ipynb_checkpoints' in model_path:
        continue
    model=MyBertModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR+'/'+model_path))
    test_pred,decodes =predict(model,test_loader,test_features,is_test=True)
    test_pred = list(set(test_pred)) #去重操作
    test_preds.append(test_pred)
    del model
    torch.cuda.empty_cache()
print('模型个数',len(os.listdir(MODEL_DIR)))


# In[77]:


VOTE_THRESHOLD = 20
number=0
all_cnt = 0
res = []
no_res_i = []
results = collections.defaultdict(int)
for model_i_predict_res in test_preds:
    for row in model_i_predict_res:
            results[row] += 1
#  票数大于预设阈值VOTE_THRESHOLD
results = [k for k,v in results.items() if v >= VOTE_THRESHOLD and len(k[4])>1 and len(k[4])<40] #
r = pd.DataFrame(results)
r.columns = ['ID','Pos_b','Pos_e','Category','Privacy']
r = r[['ID','Pos_b','Pos_e','Category','Privacy']]
r.columns = ['text_id',  'from_index', 'to_index', 'entity','word']
print('预测完毕')


# In[79]:


print('开始融合')
res1 = readResults('../submit/8989_B/results')
r = pd.concat([res1,r]).drop_duplicates()
print('融合完毕',r.shape)


# # save file

# In[82]:


def writeAns(r,path_name):
    '''写入提交文件'''
    r = r[['text_id','from_index','to_index','entity','word']]
    base_path = '../submit/' + path_name 
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for i in range(21):
        filename = base_path+'/'+str(i) + '.tag'
        with open(filename,'w+') as f:
            for i,row in r[r.text_id==i].sort_values('from_index').iterrows():
                line = row.tolist()
                line = '#'.join([str(x) for x in line[1:]]) + '\n'
                f.write(line)
writeAns(r,'results')

