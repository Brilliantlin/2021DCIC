#!/usr/bin/env python
# coding: utf-8

import re
import os
import pandas as pd
import zipfile
from tqdm.notebook import tqdm as tqdm
import numpy as np


import zipfile
def file2zip(zip_file_name: str, file_names: list):
    """ 将多个文件夹中文件压缩存储为zip
    
    :param zip_file_name:   /root/Document/test.zip
    :param file_names:      ['/root/user/doc/test.txt', ...]
    :return: 
    """
    # 读取写入方式 ZipFile requires mode 'r', 'w', 'x', or 'a'
    # 压缩方式  ZIP_STORED： 存储； ZIP_DEFLATED： 压缩存储
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in file_names:
            parent_path, name = os.path.split(fn)
            
            # zipfile 内置提供的将文件压缩存储在.zip文件中， arcname即zip文件中存入文件的名称
            # 给予的归档名为 arcname (默认情况下将与 filename 一致，但是不带驱动器盘符并会移除开头的路径分隔符)
            zf.write(fn, arcname='results/' + name)
def getSubmit(myname,df):
    print(myname)
    df = df.drop_duplicates()
    write_path1 = writeAns(df,myname+'/results') #写入文件,得到write path 的文件夹
    filenames = [write_path1 +'/'+ x for x in os.listdir(write_path1) if '.tag' in x]
    if len(filenames) != 21:
        raise ValueError('文件没有21个！')
    file2zip('../user_data/'+myname+'/results'+'.zip',filenames)
def writeAns(r,path_name):
    '''写入提交文件'''
    r = r[['text_id','from_index','to_index','entity','word']]
    base_path = '../user_data/' + path_name 
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for i in range(21):
        filename = base_path+'/'+str(i) + '.tag'
        with open(filename,'w+') as f:
            for i,row in r[r.text_id==i].sort_values('from_index').iterrows():
                line = row.tolist()
                line = '#'.join([str(x) for x in line[1:]]) + '\n'
                f.write(line)
    pd.DataFrame(r).to_csv(base_path + '.csv',index=None)
    print(base_path)
    return base_path


# In[3]:


def decorate(word, entity):
    colors = [
        'Gold', 'DarkTurquoise', 'NavajoWhite', 'Orange', 'Blue', 'Brown',
        'Peru', 'GreenYellow', 'red', 'DarkGreen'
    ]
    entitys = [
        'Tsize', 'Caps', 'Sate', 'Tloc', 'This', 'Tnum', 'Tdiff', 'LC', 'MVI',
        'TNM'
    ]
    color_map = {k: v for k, v in zip(entitys, colors)}
    color = color_map[entity]
    return "<font size = '4' color='" + color + "'>" + word + "</font>"


# In[4]:


def to_set(df):
    '''df转set'''
    df = df[['text_id',  'from_index', 'to_index', 'entity','word']].sort_values(['text_id','from_index'])
    s = []
    for i,line in df.iterrows():
        s.append(tuple(line))
    return set(s)


# In[5]:


# 用于读取预处理文本


# In[6]:


def readText(filename):
    '''读取文本'''
    with open(filename, encoding='gbk') as f:
        content = f.readlines()
    return ''.join([x.replace('\n','，，') for x in content])
#读取训练集
def readdata(text_id):
    with open('../raw_data/sample_example/%s.txt' % (text_id),encoding='gbk') as f:
        content = f.readlines()
    content = ''.join([x.replace('\n','，，') for x in content])
    ann_info = pd.read_csv('../raw_data/sample_example/%s.tag' % (text_id),header=None)
    ann_info = ann_info[0].str.split('#',expand=True)
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


# In[7]:


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
    res_path = '../user_data/' + name
    ann = []
    for file_name in os.listdir(res_path):
        if file_name=='.ipynb_checkpoints':
            continue
        text_id = file_name.split('.')[0]
        file_name = res_path + '/'+ file_name
        ann.append(readAnn(file_name,text_id))
    #     break
    return pd.concat(ann)


# # partten

# In[8]:


def findPatten(P,PATTENS,texts):
    res = [x for x in PATTENS[P].finditer(texts)]
    return res  #
corpus = '小梁型、经典型、假腺样型、紫癜型、紫癜样型、巨细胞型、破骨细胞巨细胞型、梭形细胞型、多形细胞型、透明细胞型、富含脂质型、硬化型、类黑色素型、淋巴间质型、小细胞型、低分化型、未分化型、自发坏死型、假腺管型、富脂质型、假腺样型、富于脂质型、梁索型、粗梁型、富脂型、硬化型、粗梁型、实体型、假腺型、细梁型、透明细胞型、透明细胞亚型、菊形团型、泡沫样细胞'
This_dict = corpus.replace('、','|')
PATTENS_KUOHAO = {
'Tloc-左中右肝' : re.compile(r'([左中右][左中右半、三]*肝)(肿瘤|癌|肿物|组织|、|）)'),
'Tloc-尾状叶'  : re.compile(r'肝*(尾状叶)(肿瘤|癌|肿物|组织|）)'), 
'Tloc-左右半肝' : re.compile(r'([左中右]半肝)'),
'Tloc-肝前后外叶' : re.compile(r'([左中右][肝前后内外]*叶)')
}
PATTENS = {

'MVI-1' : re.compile('[^x](M\d级*)'), 
'MVI-3' : re.compile('([查未可]*见)[^肉]?[^眼]?脉管内*癌栓[^\(（]'),
'MVI-4' : re.compile('([查未可]*见).?.?微血管内*癌栓'),
'MVI-5' : re.compile('([查未可]*见).?.?微血管内*侵犯'),
    
'TNM-1' : re.compile(r'(\w*?T\({0,1}\w\){0,1}\w*?M[x\d])'), #
  
'Tnum-1' : re.compile(r'肿[瘤物]共*(.个)') , 
'Tnum-2' : re.compile(r'肿瘤结节(\d枚)'),   
'Tnum-3' : re.compile(r'(\d枚)肿瘤结节'),  

'Tsize-1' :  re.compile(r'[^于><](\d[\.\-×~\d \*]*cm)'), 

'Tloc-4' : re.compile(r'(?<=段、)([ⅣⅡⅤⅦⅧⅥIV]+段)'), 
'Tloc-5' : re.compile(r'[^、]([ⅣⅡⅤⅦⅧⅥ\-、IV]+段)'),
'Tloc-6' : re.compile(r'([ⅣⅡⅤⅦⅧⅥ\-、IV])肿瘤'),

# s6段、s8段 联合     
'Tloc-sx、sx、sx' : re.compile(r'[肝癌物瘤]([Ss][1-9、Ss\-\.]+\d)'),
'Tloc-sx' : re.compile(r'[肝癌物瘤]([Ss]\d)[^\.\-][^S\d]'), # 过滤S100
'Tloc-9' : re.compile(r'[^、]([I-V-、]+段)'), #肝II-III段 VII段  肝IV、V段
  

'LC-早期肝硬化' : re.compile(r'(早期)肝硬化'),  # 早期肝硬化。换行 的情况2个。未考虑到
'LC-静止期' : re.compile(r'(静止期)'),
'LC-纤维化分期S（0~4期）：\d期' : re.compile(r'纤维化分期S（0~4期）：([\d\-~]{1,3}期)'), 
'LC-G1s1' : re.compile(r'[，,（\(](G\dS[\d\w\-]*)[，,）\)]') , 
'LC-纤维化n期' : re.compile(r'(肝*纤维化*[\d\-]{1,4}期)'), 

'LC-活动期' : re.compile(r'(活动期)'),
'LC-大小结节性' : re.compile(r'呈(大小结节混合性)肝硬化[^改变]{2}'),

'This-all' : re.compile(r'('+This_dict+')'),
    
'Tdiff-x级' : re.compile(r'([IVⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIV\-~]+级)'),
'Tdiff-低分化相关' : re.compile(r'[^-](低分化)[胆]'),
'Tdiff-中分化相关' : re.compile(r'[^其](中分化)[胆]'),
'Tdiff-高分化相关' : re.compile(r'[^考]?[^虑](高分化)'),
'Tdiff-中_低分化相关' : re.compile(r'([高中低]-[高中低]分化)[胆]'),
'Tdiff-中低2分化相关' : re.compile(r'[^其](中低分化)'),

# 包膜 总数246
'Caps-包膜完整不完整' : re.compile(r'包膜尚*?([不]*?完整)'), #7
# # 'Caps-未突破包膜' : re.compile(r'(未突破)包膜'), # 2 AB 🐖
'Caps-未穿破包膜' : re.compile(r'(不完整|完整|未穿破|未穿透|侵及)包膜'), # 3 AB
'Caps-未见可见肿瘤包膜' : re.compile(r'(有|无|未见|可见)明*显*肿*瘤*包膜') ,#11 👆
'Caps-侵犯包膜' : re.compile(r'(侵犯)肿*瘤*包膜'), #75 👆
'Caps-包膜侵犯' : re.compile(r'包膜伴*(侵犯)'), #3 A


'Sate-xx卫星子灶!形成' : re.compile(r'(可*见|未见|查见|多发)明*[确显]*卫星.灶(?!形成)'), #xx卫星子灶,后面没有形成 
'Sate-未见卫星子灶形成' : re.compile(r'(未见)明*[确显]*卫星.灶形成'), # 未见和形成，将未见标出 
'Sate-见卫星子灶形成' : re.compile(r'[^未见明确显]卫星.灶(形成)'), #
'Sate-可见星子灶形成' : re.compile(r'可见卫星.灶(形成)'),#
'Sate-见卫星子灶（）形成' : re.compile(r'卫星.灶（.*）(形成)'),  # 
'Sate-伴查见镜下卫星子灶' : re.compile(r'伴(查见)镜下卫星.灶'),  #
'Sate-未见.*及卫星.灶' : re.compile(r'(未见).{0,20}?[及和][^未见]{0,2}?卫星.灶'),#
'Sate-[查可]见.*及卫星.灶' : re.compile(r'([查可]见).{0,20}[及和][^查可见多发]{0,2}卫星.灶[^形成]{12,20}'),#

'Sate-伴.*及卫星.灶' : re.compile(r'伴*.{0,10}卫星.灶[及和][^查可见多发].{0,15}(形成)'), #
'Sate-卫星子灶及胆管内癌栓(形成)' : re.compile(r'卫星子灶及胆管内癌栓(形成)'), 

}  


# # extract

# In[9]:


def filtLC(df,p,flag):
    '''filt partten'''
    lc = df
    length = len(df)
    if flag:
        lc = [x for x in lc if not re.match(p,x[4])]
    return lc,length - len(lc)
def dealLC2(df,p1,p2,entity):
    '''保留l1'''
    keep = [x for x in df if x[3]!=entity]
    lc = [x for x in df if x[3]==entity]
    length = len(df)
    four = False
    huodongqi = False
    #寻找l1
    for i in lc:
        if re.match(p1,i[4]) :
            four = True

            break
    #寻找l2
    for i in lc:
        if re.match(p2,i[4]) :
            huodongqi = True
            break
    if huodongqi and four :
        lc = [x for x in lc if not re.match(p2,x[4])]
    res = keep + lc
    if len(res) != len(df):
#         print(p1)
#         print('del',set(df) - set(res))
        pass
    return res,length - len(res)
def getAns(texts,ids,split_off,pattens):
    '''抽取group里的实体'''
    df = []
    #抽取逻辑
    for P in pattens:
        res = findPatten(P,pattens,texts)
#         if res!=[]:
#             print(res)
        for line in res:
            if len(line.group()) < 1:
                continue
            b = int(line.span(1)[0])
            e = int(line.span(1)[1])
            c = P.split('-')[0]
            word = texts[b:e]
            if  len(word) < 1:
                continue
            df.append((ids, b + split_off, e + split_off, c, word))   
    return df
def filt2(df,entity):
    '''后处理，结节性优先级最低'''
    keep = [x for x in df if x[3]!=entity]
    lc = [x for x in df if x[3]==entity]
    no_jiejie = [x for x in lc if '结节' not in x[4]] 
    jiejie = [x for x in lc if '结节' in x[4]]
#     print(no_jiejie, len(no_jiejie))
    if len(no_jiejie) != 0: #如果存在其他的L
        return keep + no_jiejie
    else:
#         print('no LC append jiejie',jiejie)
        return keep + jiejie
    
def extract(texts, ids,text_id,spilt_off=0,filter_flag = False):
    '''抽取逻辑'''
    df = getAns(texts,ids,split_off,PATTENS)
    l1 = len(df)
    df = list(set(df)) #去重 
    l2 = len(df)
    if l1 != l2:
#         print('去重！'+'*' *50)
        pass
    extract_res = df
    if len(df) != 0:
        drop_cnt = 0
        cnt = 0
        # 0.2612 - 0.2692
        #LC 后处理

        df,cnt = dealLC2(df,re.compile(r'\d期'),re.compile(r'肝*纤维化\d期'),'LC') #     
        df,cnt = dealLC2(df,re.compile(r'\d期'),re.compile(r'G\dS\d'),'LC') #         
        df,cnt = dealLC2(df,re.compile(r'\d-\d期'),re.compile(r'肝*纤维化\d-\d期'),'LC') # 
        df,cnt = dealLC2(df,re.compile(r'早期'),re.compile(r'[^化]*[\d\-\d]{1,3}期'),'LC') # 
        df,cnt = dealLC2(df,re.compile(r'G\dS[\d\w\-]*'),re.compile(r'早期'),'LC') # 
        df,cnt = filtLC(df,re.compile(r'[^化维]*\d期'),filter_flag) #
        df     = filt2(df,'LC')
        df,cnt = dealLC2(df,re.compile(r'活动期'),re.compile(r'[^化]*\d期'),'LC') # 
        df,cnt = dealLC2(df,re.compile(r'形成'),re.compile(r'(见)'),'Sate') # 
        drop_cnt += cnt
        df,cnt = dealLC2(df,re.compile(r'未见'),re.compile(r'形成'),'Sate') # 
        drop_cnt += cnt
        if drop_cnt>0:
#             print(ids,text_id,extract_res,drop_cnt,df)
            pass
#         if len(df) > 1 and '、诊断：' in texts:
        if len(df) > 1 :
#             print(df,ids,text_id)
            pass
        
        df = pd.DataFrame(df,columns=['text_id',  'from_index', 'to_index', 'entity','word'])
        t = df.entity.value_counts()
    else:
        df = pd.DataFrame()

    return df

def extract_kuohao(text,i,j,split_off):
    '''抽取括号里的内容后，再对文本进行处理'''
    df = []
    ids = i
    kuohao_partten = re.compile(r'[（\(][^（]{2,20}[\)）]')
    kuohao_content = list(re.finditer(kuohao_partten,text))
    for line in kuohao_content:
        text = line.group()
        start_index = line.span()[0] + split_off
        df += getAns(text,ids,start_index,PATTENS_KUOHAO)
    if len(df) !=0:
        df = pd.DataFrame(df,columns=['text_id',  'from_index', 'to_index', 'entity','word'])
    else:
        df = pd.DataFrame()
    return df


# # run

# In[10]:


allres = []
df = []
data_path =  '../raw_data/train/'
filenamepair = {
        data_path + 'txt/' + str(i) + '.txt':
        data_path + 'label/' + str(i) + '.csv'
        for i in range(20)
    }
all_texts = ''
filter_p = re.compile('G\d-\dS\d')
for i, (text_file_name, ann_file_name) in enumerate(filenamepair.items()):  #读取所有文本数据
    filter_flag = True
    split_off = 0
    texts = readText(text_file_name)
    all_texts += texts
    if re.search(filter_p,texts): #匹配到模式，不进行过滤
#         print(re.search(filter_p,texts))
        filter_flag = False
    html = ''
    from_index = 0
    for j,text in enumerate(texts.split('，，，，')):
#         print(text)
        t = []
        t.append(extract(text,i,j,split_off,filter_flag))
        t.append(extract_kuohao(text,i,j,split_off))
        t = pd.concat(t)
        if t.shape[0]!=0:
            t = t.sort_values('from_index').drop_duplicates()
            df.append(t)
            for n, row in t.iterrows():
                html += texts[from_index:row.from_index]  #添加前面的部分
                html += decorate(row.word, row.entity)  #添加实体部分
                from_index = row.to_index  #更新起点
        split_off += len(text) + 4
    html = html.replace('，，', '<br/>')
    html_name = '../user_data/html/train_' + str(i) + '.html'
    with open(html_name, 'w+', encoding='gbk') as f:
        f.write(html)
df = pd.concat(df).drop_duplicates()
print('数据抽取完毕！',df.shape)
getSubmit('train_data_tag',df)


allres = []
df = []
data_path =  '../raw_data/test/'
filenamepair = {
        data_path + 'txt/' + str(i) + '.txt':
        data_path + 'label/' + str(i) + '.csv'
        for i in range(21)
    }
all_texts = ''
filter_p = re.compile('G\d-\dS\d')
for i, (text_file_name, ann_file_name) in enumerate(filenamepair.items()):  #读取所有文本数据
    filter_flag = True
    split_off = 0
    texts = readText(text_file_name)
    all_texts += texts
    if re.search(filter_p,texts): #匹配到模式，不进行过滤
#         print(re.search(filter_p,texts))
        filter_flag = False
    html = ''
    from_index = 0
    for j,text in enumerate(texts.split('，，，，')):
#         print(text)
        t = []
        t.append(extract(text,i,j,split_off,filter_flag))
        t.append(extract_kuohao(text,i,j,split_off))
        t = pd.concat(t)
        if t.shape[0]!=0:
            t = t.sort_values('from_index').drop_duplicates()
            df.append(t)
            for n, row in t.iterrows():
                html += texts[from_index:row.from_index]  #添加前面的部分
                html += decorate(row.word, row.entity)  #添加实体部分
                from_index = row.to_index  #更新起点
        split_off += len(text) + 4
    html = html.replace('，，', '<br/>')
    html_name = '../user_data/html/train_' + str(i) + '.html'
    with open(html_name, 'w+', encoding='gbk') as f:
        f.write(html)

df = pd.concat(df).drop_duplicates()
print(df.shape)
getSubmit('8989_B',df)

