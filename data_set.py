#-*-coding:UTF-8-*-
#actor:NJUST_Tang Bin
#@file: data_set
#@time: 2022/5/9 14:43
import torch
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import logging
from torch.nn.utils.rnn import pad_sequence
logger=logging.getLogger(__name__)

class TextClassfication(Dataset):
    '''通用文本分类数据类'''
    def __init__(self,tokenizer,max_len,data_dir,data_set_name,path_file=None,is_over_write=True):
        '''
        初始化函数
        :param tokenizer: 分词器
        :param max_len: 数据的最大长度
        :param data_dir: 保存缓存文件的路径
        :param data_set_name: 数据集名字
        :param path_file: 原始数据文件
        :param is_over_write: 是否重新生成缓存文件
        :return:
        '''
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.data_set_name=data_set_name
        cached_feature_file=os.path.join(data_dir,"cached_{}_{}".format(data_set_name,max_len))
        #判断缓存文件是否存在，如果存在，则直接加载处理后数据
        if os.path.exists(cached_feature_file) and not is_over_write:
            logger.info('已经存在缓存文件{}，直接加载'.format(cached_feature_file))
            self.data_set=torch.load(cached_feature_file)["data_set"]
        #如果缓存文件不存在，则对原始数据进行数据处理操作，并将处理后的数据存成缓存文件
        else:
            logger.info('不存在缓存文件{}，进行数据预处理操作'.format(cached_feature_file))
            self.data_set=self.load_data(path_file)
            logger.info('数据预处理操作完成，将处理后的数据存到{}中，作为缓存文件'.format(cached_feature_file))
            torch.save({'data_set':self.data_set},cached_feature_file)

    def load_data(self,path_file):
        '''
        加载原始数据，生成数据处理后的数据
        :param path_file:原始数据路径
        :return:
        '''
        self.data_set=[]
        with open(path_file,"r",encoding='utf-8') as fh:
            for idx,line in enumerate(tqdm(fh,desc='iter',disable=False)):
                sample=line.strip('\n').split('\t')
                if len(sample)==2:
                    sentence1=sample[1]
                    sentence2=None
                    label=int(sample[0])
                elif len(sample)==3:
                    sentence1=sample[1]
                    sentence2=sample[2]
                    label=int(sample[0])
                else:
                    raise ValueError("数据文件存在多列，请排查原因！")
                input_ids,token_type_ids=self.convert_feature(sentence1,sentence2)
                self.data_set.append({"text":sentence1,
                                      "input_ids":input_ids,
                                      "token_type_ids":token_type_ids,
                                      "label":label})
        return self.data_set

    def convert_feature(self,sentence1,sentence2):
        '''
        数据处理函数
        :param sample: 输入的每个文本
        :return:
        '''
        if sentence2==None:
            sentence_tokens=self.tokenizer.tokenize(sentence1)
            tokens=['[CLS]']+sentence_tokens+['[SEP]']
            input_ids=self.tokenizer.convert_tokens_to_ids(tokens)
            token_type_ids=[0]*len(input_ids)
        else:
            tokens_a=self.tokenizer.tokenize(sentence1)
            tokens_b=self.tokenizer.tokenize(sentence2)
            sentence_tokens=['[CLS]']+tokens_a+['[SEP]']+tokens_b+['[SEP]']
            input_ids=self.tokenizer.convert_tokens_to_ids(sentence_tokens)
            token_type_ids=[0]*(len(tokens_a)+2)+[1]*(len(tokens_b)+1)

        assert len(input_ids)==len(token_type_ids)
        assert len(input_ids)<=512
        return input_ids,token_type_ids

    def __len__(self):
        return len(self.data_set)
    def __getitem__(self, idx):
        instance=self.data_set[idx]
        return instance

def collate_func(batch_data):
    '''
    DataLoader所需的collate_func函数，将数据处理成tensor形式
    :param batch_data: batch数据
    :return:
    '''
    batch_size=len(batch_data)

    if batch_size==0:
        return {}
    input_ids_list,token_type_ids_list,label_list=[],[],[]
    sample_list=[]
    for instance in batch_data:
        #按照batch中的最大数据长度，对数据进行padding填充
        input_ids_temp=instance["input_ids"]
        token_type_ids_temp=instance["token_type_ids"]
        label_temp=instance["label"]
        sample={"text":instance["text"]}
        input_ids_list.append(torch.tensor(input_ids_temp,dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp,dtype=torch.long))
        label_list.append(label_temp)
        sample_list.append(sample)
    #将list中的所有tensor进行长度补全
    return {"input_ids":pad_sequence(input_ids_list,batch_first=True,padding_value=0),
            "token_type_ids":pad_sequence(token_type_ids_list,batch_first=True,padding_value=0),
            "labels":torch.tensor(label_list,dtype=torch.long),
            "samples":sample_list}

