import numpy as np
import pandas as pd
import random
import torch
import matplotlib.pylab as plt
import torchmetrics
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertModel
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial  # partial()函数可以用来固定某些参数值，并返回一个新的callable对象
import pdb
from transformers import BertForTokenClassification
from sklearn.model_selection import train_test_split

def load_data(text, tokenizer: BertTokenizer,isTest) -> dict:
    input_ids = []
    input_type_ids = []
    attention_mask = []
    label = text[2]
    encoded_inputs = tokenizer.encode_plus(text[0], text[1],
                                           add_special_tokens=True, return_token_type_ids=True,
                                           return_attention_mask=True,padding=True)

    if isTest == True:
        return encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], \
           encoded_inputs["attention_mask"]
    else:
        return encoded_inputs["input_ids"], encoded_inputs["token_type_ids"], \
           encoded_inputs["attention_mask"], label


# 通过词典导入分词器
MODEL_PATH = './dataset/model/'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
# b. 导入配置文件
# model_config = BertConfig.from_pretrained("./dataset/model/bert-base-chinese/")
# 对训练集和测试集进行编码


# functools.partial()的功能：预先设置参数，减少使用时设置的参数个数
# 使用partial()来固定convert_example函数的tokenizer, label_vocab, max_seq_length等参数值
trans_func = partial(load_data, tokenizer=tokenizer,isTest=False)
trans_func_test = partial(load_data, tokenizer=tokenizer,isTest=True)
"""
1.将现有数据转为token 就是文字转字典类型
"""
train_df = pd.read_csv("dataset/train.txt", header=None, sep="\t")
train_df.insert(loc=train_df.shape[1],column=None,value=None,allow_duplicates=True)
train_df=train_df.values
train_df=train_df[:1]
for index, text in enumerate(train_df):
    train_df[index] = trans_func(text)
train_df,dev_df = train_test_split(train_df,test_size=0.3)

test_df = pd.read_csv("dataset/test.txt", header=None, sep="\t")
test_df.insert(loc=test_df.shape[1],column=None,value=None,allow_duplicates=True)
test_df=test_df.values
for index, text in enumerate(test_df):
    test_df[index] = trans_func_test(text)


"""
2.创建批量batch
"""
def create_batch(batch_data):
    tokens_tensors=[torch.tensor(s[0]) for s in batch_data]
    mask_tensors=[torch.tensor(s[2]) for s in batch_data]
    label_tensors=None
    if batch_data[0][3] is None:
        label_tensors = [torch.tensor(0) for s in batch_data]
    else:
        # label_tensors = [torch.tensor(s[3]) for s in batch_data]
        label_tensors = torch.tensor([s[3] for s in batch_data])
    one=[0]
    tokens_tensors=pad_sequence(tokens_tensors,batch_first=True)
    tokens_tensors=torch.tensor([t+one for t in tokens_tensors.numpy().tolist()])

    mask_tensors=pad_sequence(mask_tensors,batch_first=True)
    mask_tensors =torch.tensor([t+one for t in mask_tensors.numpy().tolist()])

    # label_tensors=pad_sequence(label_tensors,batch_first=False)
    # label_tensors =torch.tensor([t+one for t in label_tensors.numpy().tolist()])

    return tokens_tensors,mask_tensors,label_tensors


trainloader=DataLoader(train_df,batch_size=32,collate_fn=create_batch,drop_last=False)
dev_df=DataLoader(dev_df,batch_size=32,collate_fn=create_batch,drop_last=False)
testloader=DataLoader(test_df,batch_size=32,collate_fn=create_batch,drop_last=False)



"""
进行训练
"""
model=BertForSequenceClassification.from_pretrained(MODEL_PATH)
#创建一些常规指标参数
model_recall = torchmetrics.Recall(average='macro', num_classes=2)
model_precision = torchmetrics.Precision(average='macro', num_classes=2)
model_f1 = torchmetrics.F1Score(average="macro", num_classes=2)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss(ignore_index=-1)
# 在Adam的基础上加入了权重衰减的优化器，可以解决L2正则化失效问题
optimizer = torch.optim.Adam(lr=2e-5, params=model.parameters())
#看是否用cpu或者gpu训练
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")


global_step=0
total_loss = 0.0


# 评估函数
def evaluate(model, data_loader):
    # 依次处理每批数据
    for tokens_tensors,mask_tensors,label_tensor in data_loader:
        # 单字属于不同标签的概率
        output = model(input_ids=tokens_tensors.to(device), attention_mask=mask_tensors.to(device)
                       , labels=label_tensor.to(device))
        # 损失函数的平均值
        loss = output[0]
        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始
        preds=torch.argmax(output[1].detach().cpu(),dim=-1)

        model_f1.update(preds.flatten(), label_tensor.flatten())
        model_recall.update(preds.flatten(), label_tensor.flatten())
        model_precision.update(preds.flatten(), label_tensor.flatten())
    f1_score = model_f1.compute()
    recall = model_recall.compute()
    precision = model_precision.compute()

    # 清空计算对象
    model_precision.reset()
    model_f1.reset()
    model_recall.reset()
    print("评估准确度: %.6f - 召回率: %.6f - f1得分: %.6f- 损失函数: %.6f" % (precision, recall, f1_score, total_loss))


for epoch in range(1):

    #以此处理每批数据
    for step,(tokens_tensors,mask_tensors,label_tensor) in enumerate(trainloader):

        # 梯度置零
        optimizer.zero_grad()

        #进行输出
        output=model(input_ids=tokens_tensors.to(device),attention_mask = mask_tensors.to(device)
                     ,labels =label_tensor.to(device))

        preds=torch.argmax(output[1].detach().cpu(),dim=-1)

        loss=output[0]
        loss.backward()


        #根据梯度来更新参数
        optimizer.step()

        global_step+=1
        total_loss += loss.item()


        recall=model_recall(preds.flatten(), label_tensor.flatten())
        precision=model_precision(preds.flatten(), label_tensor.flatten())
        f1_score = model_f1(preds.flatten(), label_tensor.flatten())

        # 损失函数的平均值
        if global_step % 10 == 0:
            print("训练集的当前epoch:%d - step:%d" % (epoch, step))
            print("训练准确度: %.6f, 召回率: %.6f, f1得分: %.6f- 损失函数: %.6f" % (precision, recall, f1_score, loss))
            # print("训练准确度: %.6f, 召回率: %.6f, f1得分: %.6f" % (precision, recall,  f1_score))


    # 计算一个epoch的accuray、recall、precision
    total_recall = model_recall.compute()
    total_precision = model_precision.compute()
    total_f1 = model_precision.compute()

    # 清空计算对象
    model_precision.reset()
    model_f1.reset()
    model_recall.reset()

    # 评估训练模型
    evaluate(model, dev_df)
    torch.save(model.state_dict(),
               "./checkpoint/model_%d.pdparams"% (global_step))

# 模型存储
# !mkdir bert_result
model.save_pretrained('./bert_result')
tokenizer.save_pretrained('./bert_result')