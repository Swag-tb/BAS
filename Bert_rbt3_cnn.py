#-*-coding:UTF-8-*-
#actor:NJUST_Tang Bin
#@file: Bert_rbt3_cnn
#@time: 2022/6/16 15:13
import torch.nn as nn
from transformers import BertModel,BertConfig,BertLayer
import torch.nn.functional as F
import torch

class BRBT3CNNModel(nn.Module):
    def __init__(self,config):
        super(BRBT3CNNModel,self).__init__()
        self.bert=BertModel.from_pretrained(config.bert_path)
        self.rbt3=BertLayer(BertConfig.from_pretrained(config.rbt3))
        self.convs=nn.ModuleList(
            [nn.Conv2d(1,config.num_filters,(k,config.hidden_size)) for k in config.filter_size]
        )
        self.dropout=nn.Dropout(config.dropout)
        self.num_classes=config.num_classes
        self.fc=nn.Linear(config.num_filters*len(config.filter_size),self.num_classes)
        self.softmax=nn.Softmax()

    def conv_and_pool(self,x,conv):
        x=conv(x)
        x=F.relu(x)
        x=x.squeeze(3)
        x=F.max_pool1d(x,x.size(2))
        x=x.squeeze(2)
        return x

    def forward(self,input_ids=None,token_type_ids=None,labels=None):
        encoder_out=self.bert.forward(input_ids=input_ids,
                                      token_type_ids=token_type_ids)
        encoder_out=self.rbt3.forward(encoder_out[0])
        out=encoder_out[0].squeeze(1)
        out=torch.cat([self.conv_and_pool(out,conv) for conv in self.convs],1)
        out=self.fc(out)
        prediction_scores=self.softmax(out)
        prediction_label=torch.argmax(prediction_scores,dim=1)
        outputs=(prediction_scores,prediction_label,)
        if labels is not None:
            loss=F.cross_entropy(out,labels)
            outputs=(loss,)+outputs
        return outputs

import torch
import os
import json
import random
import numpy as np
import argparse
import logging
from transformers import BertTokenizer
from data_set import TextClassfication,collate_func
from torch.utils.data import DataLoader
from transformers import AdamW,get_linear_schedule_with_warmup
from tqdm import tqdm,trange
from sklearn.metrics import f1_score

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(name)s-%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger=logging.getLogger(__name__)

def train(model,device,tokenizer,args):
    if args.gradient_accumulation_steps<1:
        raise ValueError('?????????????????????????????????????????????1')
    train_batch_size=int(args.train_batch_size/args.gradient_accumulation_steps)
    train_data=TextClassfication(tokenizer,args.max_len,args.data_dir,"train_text_classification",path_file=args.train_file_path)
    train_data_loader=DataLoader(train_data,
                                 batch_size=train_batch_size,
                                 collate_fn=collate_func,
                                 shuffle=True)
    total_steps=int(len(train_data_loader)*args.num_train_epochs/args.gradient_accumulation_steps)

    dev_data=TextClassfication(tokenizer,args.max_len,args.data_dir,"dev_text_classfication",path_file=args.dev_file_path)
    test_data = TextClassfication(tokenizer, args.max_len, args.data_dir, "test_text_classfication",path_file=args.test_file_path)
    logging.info("?????????????????????{}".format(total_steps))
    model.to(device)
    #????????????????????????????????????????????????????????????

    bert_param = list((n, p) for n, p in model.named_parameters() if "bert." in n or "rbt3." in n)
    cnn_param = list((n, p) for n, p in model.named_parameters() if "convs." in n or "fc." in n)
    need_decay=args.need_decay

    optimizer_grouped_parameters=[
        {'params':[p for n,p in bert_param if any(nd in n for nd in need_decay)],
         'weight':0.01,"lr":args.bert_lr},
        {'params':[p for n,p in bert_param if not any(nd in n for nd in need_decay)],
         'weight':0.0},
        {'params': [p for n, p in cnn_param if any(nd in n for nd in need_decay)],
         'weight': 0.01, "lr": args.cnn_lr}
    ]
    for name,param in model.named_parameters():
        param.requires_grad=True
        if not any(s in name for s in need_decay):
            param.requires_grad=False

    require_grad_param=[]
    for name,param in model.named_parameters():
        if param.requires_grad:
            require_grad_param.append(name)
            print('????????????????????????:{}??????????????????{}'.format(name,param.size()))
    #???????????????
    optimizer=AdamW(optimizer_grouped_parameters,eps=args.adam_epsilon)
    schedular=get_linear_schedule_with_warmup(optimizer,
                                              num_warmup_steps=int(args.warmup_proportion*total_steps),
                                              num_training_steps=total_steps)

    #??????cuda??????
    torch.cuda.empty_cache()
    model.train()
    tr_loss,logging_loss=0.0,0.0
    global_step=0
    best_score=0
    for iepoch in trange(0,int(args.num_train_epochs),desc="Epoch",disable=False):
        iter_bar=tqdm(train_data_loader,desc="Iter (loss=X.XXX)",disable=False)
        for step,batch in enumerate(iter_bar):
            input_ids=batch["input_ids"].to(device)
            token_type_ids=batch["token_type_ids"].to(device)
            labels=batch["labels"].to(device)
            outputs=model.forward(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  labels=labels)
            loss=outputs[0]
            tr_loss+=loss.item()
            iter_bar.set_description("Iter (loss=%5.3f)"%loss.item())
            #?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            if args.gradient_accumulation_steps>1:
                loss=loss/args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),args.max_grad_norm)
            #???????????????????????????????????????????????????
            if (step+1)%args.gradient_accumulation_steps==0:
                optimizer.step()
                schedular.step()
                optimizer.zero_grad()
                global_step+=1
        #??????????????????????????????
        eval_acc,eval_f1,json_data=evaluate(model,device,dev_data,args)
        model.train()
        if eval_f1>best_score:
            logger.info("epoch:{}".format(iepoch))
            logger.info("dev_acc:{}".format(eval_acc))
            logger.info("dev_f1:{}".format(eval_f1))

            output_dir=args.output_dir
            model_to_save={k:v for k,v in model.named_parameters() if k in require_grad_param}
            torch.save(model_to_save,output_dir+"/pytorch_model.bin")
            # model_to_save=model.module if hasattr(model,"module") else model
            # model_to_save.save_pretrained(output_dir)

            json_output_dir=os.path.join(output_dir,"json_data.json")
            fin=open(json_output_dir,'w',encoding='utf-8')
            fin.write(json.dumps(json_data,ensure_ascii=False,indent=4))
            fin.close()
            test_acc,test_f1,test_json_data=evaluate(model,device,test_data,args)
            model.train()
            logger.info("epoch:{}".format(iepoch))
            logger.info("test_acc:{}".format(test_acc))
            logger.info("test_acc:{}".format(test_f1))
            json_output_dir=os.path.join(output_dir,"test_json_data.json")
            fin=open(json_output_dir,"w",encoding='utf-8')
            fin.write(json.dumps(test_json_data,ensure_ascii=False,indent=4))
            fin.close()
            best_score=eval_f1
        else:
            continue
        torch.cuda.empty_cache()


def evaluate(model,device,dev_data,args):
    test_data_loader=DataLoader(dev_data,batch_size=args.test_batch_size,collate_fn=collate_func,shuffle=False)
    iter_bar=tqdm(test_data_loader,desc="iter",disable=False)
    y_true=[]
    y_predict=[]
    y_scores=[]
    samples=[]
    for step,batch in enumerate(iter_bar):
        model.eval()
        with torch.no_grad():
            labels=batch["labels"]
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            scores,prediction=model.forward(input_ids=input_ids,
                                            token_type_ids=token_type_ids)
            y_true.extend(labels.numpy().tolist())
            y_predict.extend(prediction.cpu().numpy().tolist())
            y_scores.extend(scores.cpu().numpy().tolist())
            samples.extend(batch["samples"])
    json_data={"data":[],"acc":None}
    for label,pre,score,sample in zip(y_true,y_predict,y_scores,samples):
        sample["label"]=label
        sample["pre"]=pre
        sample["scores"]=score
        json_data["data"].append(sample)
    y_true=np.array(y_true)
    y_predict=np.array(y_predict)
    eval_acc=np.mean((y_true==y_predict))
    eval_f1=f1_score(y_true,y_predict,average="macro")
    class_f1=f1_score(y_true,y_predict,average=None)
    json_data["acc"]=str(eval_acc)
    json_data["f1"]=str(class_f1)
    return eval_acc,eval_f1,json_data

def set_args():
    parser=argparse.ArgumentParser()#?????????????????????
    parser.add_argument('--device',default='-1',type=str,help='???????????????????????????????????????')
    parser.add_argument('--train_file_path',default='data/train.txt',type=str,help='????????????')
    parser.add_argument('--dev_file_path', default='data/dev.txt', type=str, help='????????????')
    parser.add_argument('--test_file_path', default='data/test.txt', type=str, help='????????????')
    parser.add_argument('--output_dir', default='model_output/', type=str, help='??????????????????')
    parser.add_argument('--vocab_path', default='pre_train_model/sci-uncased/vocab.txt', type=str, help='???????????????????????????')
    parser.add_argument('--bert_path', default='D:/??????/??????/torch_pretrain_models/??????/bert_wwm_ext_chinese', type=str, help='?????????????????????')
    parser.add_argument('--rbt3', default=r'D:\??????\??????\torch_pretrain_models\??????\rbt3', type=str,help='rbt3?????????????????????')
    parser.add_argument('--data_dir', default='cached/', type=str, help='???????????????????????????')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='?????????????????????')
    parser.add_argument('--train_batch_size', default=128, type=int, help='???????????????batch?????????')
    parser.add_argument('--test_batch_size', default=128, type=int, help='???????????????batch?????????')
    parser.add_argument('--bert_lr', default=1e-5, type=float, help='?????????')
    parser.add_argument('--cnn_lr', default=1e-3, type=float, help='?????????')
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help='warmup??????????????????????????????????????????????????????warmup')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Adam????????????epsilon???')
    parser.add_argument('--gradient_accumulation_steps', default=32, type=int, help='????????????')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--seed', default=2022, type=int, help='????????????')
    parser.add_argument('--max_len', default=512, type=int, help='????????????????????????????????????')
    parser.add_argument('--num_classes', default=2, type=int, help='???????????????')
    parser.add_argument('--num_filters', default=256, help='num_filters')
    parser.add_argument('--filter_size', default=(2,3), help='num_filters')
    parser.add_argument('--hidden_size', default=768, help='hidden_size')
    parser.add_argument('--dropout', default=0.1, help='dropout')
    parser.add_argument('--need_decay', default=["bert.","rbt3.","convs.","fc."], help='need_decay')
    return parser.parse_args()#??????parse_args??????????????????

def main():
    args=set_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    device=torch.device("cuda" if torch.cuda.is_available() and int(args.device)>=0 else "cpu")
    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    model=BRBT3CNNModel(args)

    #?????????tokenizer
    tokenizer=BertTokenizer.from_pretrained(args.vocab_path,do_lower_case=True)
    #???????????????????????????
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    #????????????
    train(model,device,tokenizer,args)

if __name__=="__main__":
    main()