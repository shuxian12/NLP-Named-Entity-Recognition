import torch
from io import open
import glob
import os
import re
import preprocessor as p
import numpy as np
p.set_options(p.OPT.HASHTAG, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.URL, p.OPT.SMILEY)
regex = re.compile("(\$(HASHTAG|MENTION|RESERVED|URL|SMILEY)\$)|(\$\d+(K|k*))")

"""""""""
加載及處理word_embedding模型
"""""""""

"""
前處理 查看資料集狀況
"""
# 加載word_embedding模型
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'glove.twitter.27B.100d.txt'
word2vec_output_file = 'glove.twitter.27B.100d.word2vec.txt'

# glove_input_file = 'glove.840B.300d.txt'
# word2vec_output_file = 'glove.840B.300d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
glove_model.save('glove.twitter.27B.100d.model')
glove_model = KeyedVectors.load('glove.twitter.27B.100d.model')

"""
word_embedding矩陣 預處理 及 匯入
"""
vocab,embeddings = [],[]
order = 0
vocab_dict = {}
with open('glove.twitter.27B.100d.txt','rt') as fi:
    full_content = fi.read().strip().split('\n')
for i in range(len(full_content)):
    i_word = full_content[i].split(' ')[0]
    i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
    if i_word not in vocab_dict:
        vocab_dict[i_word] = order
        order += 1
    vocab.append(i_word)
    embeddings.append(i_embeddings)

import numpy as np
vocab_npa = np.array(vocab)
embs_npa = np.array(embeddings)

#insert '<pad>' and '<unk>' tokens at start of vocab_npa.
vocab_npa = np.append(vocab_npa, '<pad>')
vocab_npa = np.append(vocab_npa, '<unk>')
vocab_npa = np.append(vocab_npa, '<STA>')
vocab_npa = np.append(vocab_npa, '<END>')
print(vocab_npa[-10:])

pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.
sta_emb_npa = np.arange(0, 1, 0.01, dtype = "float64")   
end_emb_npa = np.arange(-1, 0, 0.01, dtype = "float64")

#insert embeddings for pad and unk tokens at top of embs_npa.
embs_npa = np.vstack((embs_npa,pad_emb_npa,unk_emb_npa,sta_emb_npa,end_emb_npa))
print(embs_npa.shape)

vocab_dict['<pad>'] = order
vocab_dict['<unk>'] = order+1
vocab_dict['<STA>'] = order+2
vocab_dict['<END>'] = order+3

with open('/content/drive/MyDrive/Colab Notebooks/NLP-1/vocab_npa.npy','wb') as f:
    np.save(f,vocab_npa)

with open('/content/drive/MyDrive/Colab Notebooks/NLP-1/embs_npa.npy','wb') as f:
    np.save(f,embs_npa)

"""
匯入詞向量矩陣(最終使用的詞向量)
"""
embs_npy = np.load('embs_npa.npy')
vocab_npy = np.load('vocab_npa.npy')

# print(vocab_npy[-10:])
print(embs_npy[-2 ])
embs_npy.shape
# vocab_dict['<END>']

"""
單字dictionary預處理及匯入
"""
import pickle
def pet_save(path, vocab_dict : dict):
    with open(path + '.pickle', 'wb') as f:
        pickle.dump(vocab_dict, f, pickle.HIGHEST_PROTOCOL)

def digimon_load(path):
    with open(path + '.pickle', 'rb') as f:
        return pickle.load(f)

path = 'vocab_dict'
vocab_dict = digimon_load(path)
# pet_save(path, vocab_dict)

"""
未知符號紀錄
"""
# "\.{2,}(\"|\')*" "..."
# "haven't" "havenot"
# "&lt;"    "punctuation"
# ".."     "..."
# "URL'"    "URL"
# ":'"       ":"
# "......"   "..."
# "&gt;"    "punctuation"
# "...:"     "..."
# "i'm"     "iam"
# "i'll"     "Ill"
# "...\""    "..."
# ".\""      "."
# "!:"       "!"
# "'I"       "I"
# "Can't"   "Cant"
# "won't"    "wont"
# "'RT"    "RESERVED"
# "w/"         "with"
# ":/"      ":"
# "...'"     "..."
# "we'll"   "wewill"
# "ain't"    "arent"
# "You're"   "you're"


"""""""""
資料集 前處理 及 匯入
"""""""""

def find_files(path): return glob.glob(path)

def textprocessor(word):
    if re.search("\$HASHTAG\$",word):
        word = re.sub("\$HASHTAG\$","<hashtag>", word)
    if re.search("\$MENTION\$",word):
        word = re.sub("\$MENTION\$","<user>", word)
    if re.search("\$RESERVED\$",word):
        word = re.sub("\$RESERVED\$","rt", word)
    if re.search("\$URL\$",word):
        word = re.sub("\$URL\$","<url>", word)
    if re.search("\$SMILEY\$",word):
        word = re.sub("\$SMILEY\$","<smile>", word)
    if re.search("\$\d+(K|k*)",word):
        word = re.sub("\$\d+(K|k*)", "price",word)
    return word

def error_text(word):
    if re.search("\.{2,}(\"|\')*", word):  
        word = re.sub("\.{2,}(\"|\'|:|\!)*", "…",word)
    elif re.search("haven't", word):
        word = re.sub("haven't", "havnt", word)
    elif re.search("(&lt;)|(&gt;)", word):
        word = re.sub("(&lt;)|(&gt;)", "punctuation", word)
    elif re.search("URL'", word):
        word = re.sub("URL'", "URL", word)
    elif re.search(":\'", word):
        word = re.sub(":\'", ":", word)
    elif re.search("i'(m|ll)", word):
        word = re.sub("i'(m|ll)", "iam", word)
    elif re.search("\.\"", word):
        word = re.sub("\.\"", ".", word)
    elif re.search("!(:|\.+)", word):
        word = re.sub("!(:|\.+)", "!", word)
    elif re.search("'I", word):
        word = re.sub("'I", "I", word)
    elif re.search("'Can't'", word):
        word = re.sub("'Can't", "cant", word)
    elif re.search("'won't'", word):
        word = re.sub("'won't", "wont", word)
    elif re.search("'RT'", word):
        word = re.sub("'RT", "rt", word)
    elif re.search("w/", word):
        word = re.sub("w/", "with", word)
    elif re.search(":/", word):
        word = re.sub(":/", ":", word)
    elif re.search("w/", word):
        word = re.sub("w/", "with", word)
    elif re.search("we'll", word):
        word = re.sub("we'll", "wewill", word)
    elif re.search("ain't", word):
        word = re.sub("ain't", "arent", word)
    elif re.search("You're", word):
        word = re.sub("You're", "youre", word)
    else:
        word = "<unk>"
    return word

tag_dict = {'O': 0, 'B-geo-loc': 1, 'B-facility': 2, 'B-product': 3, 
            'B-movie': 4, 'B-musicartist': 5, 'B-company': 6, 'B-sportsteam': 7, 
            'B-person': 8, 'B-tvshow': 9, 'B-other': 10, 'I-geo-loc': 11, 
            'I-facility': 12, 'I-product': 13, 'I-movie': 14, 
            'I-musicartist': 15, 'I-company': 16, 'I-sportsteam': 17, 
            'I-person': 18, 'I-tvshow': 19, 'I-other': 20}

tag_dict_list = ['O', 'B-geo-loc', 'B-facility', 'B-product', 'B-movie', 
                 'B-musicartist', 'B-company', 'B-sportsteam', 'B-person', 
                 'B-tvshow', 'B-other', 'I-geo-loc', 'I-facility', 'I-product',
                 'I-movie', 'I-musicartist', 'I-company', 'I-sportsteam', 
                 'I-person', 'I-tvshow', 'I-other']
test_dict = {}

def pad2mask(t):
    if t == '<pad>': #轉換成mask所用的0
        return 0
    else:
        return 1

def load_data(path, MAX_LEN = 41, train=True):
    word_list, tag_list, mask_list, tmp1, tmp2 = [], [], [], [], []
    word_num = 0  
    count = 0
    # tag_dict = {}
    
    with open(path , 'r', encoding= 'utf8') as file:
        for line in file.readlines():
            if line == "\n": 
                if train and len(tmp1) < 5:     #訓練集太短的刪掉
                    tmp1.clear(), tmp2.clear()
                    continue

                count+=1    #句子數量
                tmp1.append("<END>")
                if train:   tmp2.append(0) 
                ## padding 處理
                if len(tmp1) < MAX_LEN:   
                    pad_len = MAX_LEN - len(tmp1)
                    tmp1 = tmp1 + ['<pad>']*pad_len
                    if train:   tmp2 = tmp2 + [0]*pad_len
                elif len(tmp1) > MAX_LEN:  
                    tmp1 = tmp1[:MAX_LEN-1]
                    tmp1.append("<END>")
                    if train:
                        tmp2 = tmp2[:MAX_LEN-1]
                        tmp2.append(0)
                mask_list.append([pad2mask(t) for t in tmp1])
                word_list.append([vocab_dict[t] for t in tmp1])
                tmp1.clear(), tmp1.append("<STA>")
                if train:   
                    tag_list.append(tmp2.copy())
                    tmp2.clear(), tmp2.append(0) 
                continue

            # print(line)
            if train:
                name, tag = line.split('\t')
                tag = tag.strip('\n')
            else:
                name = line.strip('\n')
                # if word_num < 10: print(name)

            if(name[0] == '#'):
                name = name[1:]
            name = p.tokenize(name)
            name = name.lower()
            if regex.search(name): 
                name = textprocessor(name)
                
            # if word_num < 10 and not train: print(name), print(vocab_dict[name])
            try:
                vocab_dict[name]
            except KeyError:
                name = error_text(name)
                try:
                    vocab_dict[name]
                except KeyError:
                    name = '<unk>'
            word_num += 1
            tmp1.append(name)  #tweet-preprocessor
            if train:   tmp2.append(tag_dict[tag])
            # if tag not in tag_dict:
            #     tag_dict[tag] = 1
            # else: tag_dict[tag] += 1
        
    print(count)
    word_list = torch.LongTensor(word_list)
    tag_list = torch.LongTensor(tag_list)
    mask_list = torch.tensor(mask_list, dtype=torch.uint8)
    print(len(word_list[0]))
    print(mask_list.shape)
    return word_list, tag_list, mask_list, word_num


"""
資料匯入
"""
train_path = 'train.txt'
dev_path = 'dev.txt'
train_list ,train_tag_list, train_mask_list, train_num_word = load_data(train_path)
dev_list ,dev_tag_list, dev_mask_list, dev_num_word = load_data(dev_path)

### 確認單字集正確
# k = 0
# a = vocab_dict['<STA>']
# b = vocab_dict['<END>']
# c = vocab_dict['<pad>']
# for i in range(len(train_list)):
#     for j in range(len(train_list[i])):
#         if train_list[i][j] != a and train_list[i][j] != b and train_list[i][j] != c:
#             k+=1

"""
資料集未知字串處理及計算，正式處理前使用的
"""
# count = 0
# error_dic = {}
# for line in train_list:
#   for word in line:
#     try:
#         vec = glove_model[word]
#         # print(cat_vec)
#         # print(glove_model.most_similar(s))
#         # embedding_matrix[i] = embedding_vector
#     except KeyError:
#         # print("error: " + word)
#         count+=1
#         if word in error_dic: error_dic[word]+=1
#         else: error_dic[word] = 1
# num = 0
# key = ''
# print(count)
# for word in error_dic:
#   if error_dic[word] > 5:  print(word + ':' , error_dic[word])
#   if num < error_dic[word]: 
#     num = error_dic[word]
#     key = word

# print(num, " "+key)
# #處理文字使用者名字tag還有標點符號和網址
# #處理pad unknown


"""""""""
模型建立
"""""""""

"""
Dateset
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class WordDataset(Dataset):
    def __init__(self, word_list, mask_list, tag_list, test = False):
      self.word_list = word_list
      self.tag_list = tag_list
      self.mask_list = mask_list
      self.test = test
    #   self.nums = len(self.word_list)

    def __len__(self):
      return len(self.word_list)

    def __getitem__(self, item):
      if self.test: return self.word_list[item], self.mask_list[item]
      return self.word_list[item], self.tag_list[item], self.mask_list[item]


"""
bilstm-crf 模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
import math
from sklearn.metrics import f1_score
torch.manual_seed(1)


class BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, embs_vec, num_layer, dropout):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim  #100
        self.hidden_dim = hidden_dim
        self.tagset_size = 21       # len(tag_to_ix)
        self.batch_size = batch_size
        self.embs_vec = embs_vec    # embs_npa
        self.num_layer = num_layer
        
        self.drop = nn.Dropout(dropout)

        #embedding
        self.word_embeds = nn.Embedding.from_pretrained(torch.from_numpy(self.embs_vec).float())
        # self.word_embeds = nn.Embedding(vocab_size,embedding_dim,padding_idx=self.pad_idx)  
        
        #lstm
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers = self.num_layer, bidirectional = True)
        
        #LSTM的輸出對應tag空間（tag space）
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)  
        #輸入是[batch_size, size]中的size，輸出是[batch_size，output_size]的output_size
        
        #CRF
        self.crf = CRF(self.tagset_size)   #default batch_first=False
 
   
    def forward(self, sentence, tags=None, mask=None):     #sentence=(batch,seq_len)   tags=(batch,seq_len)
        self.batch_size = sentence.shape[0]   #防止最後一batch中的數量不夠原本的BATCH_SIZE
        # print('batch_size: ',self.batch_size)

        # embedding
        embeds = self.word_embeds(sentence).permute(1,0,2)   #output=[seq_len, batch_size, embedding_size]
        # embeds = self.drop(embeds)

        # bilstm
        if use_gpu:
            self.hidden = (torch.randn(2,self.batch_size,self.hidden_dim//2).cuda(),torch.randn(2,self.batch_size,self.hidden_dim//2).cuda())
        else: self.hidden = (torch.randn(2,self.batch_size,self.hidden_dim//2),torch.randn(2,self.batch_size,self.hidden_dim//2)) 

        lstm_out, hidden = self.lstm(embeds, self.hidden)
        lstm_out = self.drop(lstm_out)

        # full
        # 從lstm的輸出轉為tagset_size長度的向量組（即輸出了每個tag的可能性）
        lstm_feats = self.hidden2tag(lstm_out)
        
        #crf 
        if tags is not None: #train
            if mask is not None:
                loss = -1.*self.crf(emissions=lstm_feats,tags=tags.permute(1,0),mask=mask.permute(1,0),reduction='mean')   #outputs=(batch_size,)   輸出log形式的likelihood
                prediction = self.crf.decode(emissions=lstm_feats,mask=mask.permute(1,0))  
            else:
                loss = -1.*self.crf(emissions=lstm_feats,tags=tags.permute(1,0),reduction='mean')
                prediction = self.crf.decode(emissions=lstm_feats)
            return loss, prediction
        else:   #test
            if mask is not None:
                prediction = self.crf.decode(emissions=lstm_feats,mask=mask.permute(1,0))   
            else:
                prediction = self.crf.decode(emissions=lstm_feats)
            return prediction

#batch_masks:tensor數據，結構為(batch_size,MAX_LEN) 
#batch_labels: tensor數據，結構為(batch_size,MAX_LEN) 
#batch_prediction:list數據，結構為(batch_size,) #每個數據長度不一（在model參數mask存在的情況下）

def f1_score_evaluation(batch_masks, batch_labels, batch_prediction):
    batch_masks = batch_masks.cpu()
    batch_labels = batch_labels.cpu()
    # batch_prediction = batch_prediction.cpu()
    all_prediction = []
    all_labels = []
    batch_size = batch_masks.shape[0]   #防止batch_size不夠

    for index in range(batch_size):
        #把沒有mask掉的原始tag都集合到一起
        length = sum(batch_masks[index].numpy()==1)
        _label = batch_labels[index].numpy().tolist()[:length]
        all_labels = all_labels+_label

        #把沒有mask掉的預測tag都集合到一起
        all_prediction = all_prediction+batch_prediction[index]
        
        assert len(_label)==len(batch_prediction[index])
  
        
    assert len(all_prediction) == len(all_labels)
    score = f1_score(all_prediction,all_labels,average='weighted')
    
    return score, all_prediction, all_labels

def test_f1_score_evaluation(batch_masks, batch_prediction, batch_words, batch_labels = None):
    # print(len(batch_labels), " ", len(batch_prediction))
    batch_masks = batch_masks.cpu()
    if batch_labels is not None:     batch_labels = batch_labels.cpu()
    # batch_prediction = batch_prediction.cpu()
    all_prediction, all_labels = [], []
    test_pre, test_label, test_word = [], [], []
    batch_size = batch_masks.shape[0]   #防止最後一batch的數據不夠batch_size

    if batch_labels is not None:
        for index in range(batch_size):
            # 把沒有mask掉的原始tag都集合到一起
            length = sum(batch_masks[index].numpy()==1)
            _label = batch_labels[index].numpy().tolist()[:length]
            all_labels = all_labels+_label
            test_label += _label

            #把沒有mask掉的預測tag都集合到一起
            all_prediction = all_prediction+batch_prediction[index]
            test_pre += batch_prediction[index]
            test_word += batch_words[index].numpy().tolist()[:length]
            
            assert len(_label)==len(batch_prediction[index])
    
            
        assert len(all_prediction) == len(all_labels)
        score = f1_score(all_prediction,all_labels,average='weighted')
        return score, test_pre, test_label, test_word
    else:
        for index in range(batch_size):
            length = sum(batch_masks[index].numpy()==1)

            #把沒有mask掉的預測tag都集合到一起
            all_prediction = all_prediction+batch_prediction[index]
            test_pre += batch_prediction[index]
            test_word += batch_words[index].numpy().tolist()[:length]
        return test_pre, test_word

"""""""""
=============== !!!!!!!!!!!!!! 參數設定  !!!!!!!!!!!!!! ==============
"""""""""
HIDDEN_SIZE = 1024*2
EMBEDDING_DIM = 100
BATCH_SIZE = 4
NUM_LAYERS = 1
DROPOUT = 0.5
use_gpu = False #torch.cuda.is_available()
"""""""""
=============== !!!!!!!!!!!!!! 參數設定 !!!!!!!!!!!!!! ==============
"""""""""


"""
train and dev 設定
"""
samples_cnt = train_list.shape[0]
batch_cnt = math.ceil(samples_cnt/BATCH_SIZE)   
    
def train(epoch, model, optimizer):
    model.train()
    # if use_gpu:
    #     model.cuda()

    for step, (word_list, tag_list, mask_list) in enumerate(train_loader):
        model.zero_grad()
        if use_gpu:
            word_list = word_list.cuda()
            tag_list = tag_list.cuda()
            mask_list = mask_list.cuda()

        loss, _ = model(word_list, tag_list, mask_list) 
        if step%100 ==0:
            print('Epoch=%d  step=%d/%d  loss=%.5f' % (epoch,step, batch_cnt, loss))
                                         
        loss.backward()      
        optimizer.step()  
    return dev(model, optimizer)

def dev(model, optimizer):
    model.eval()
    score_list, pre_list, label_list = [], [], []
    n ,hit = 0, 0
    # if use_gpu:
    #     model.cuda()

    with torch.no_grad():
        for step, (word_list, tag_list, mask_list) in enumerate(dev_loader): 
            if use_gpu:
                word_list = word_list.cuda()
                tag_list = tag_list.cuda()
                mask_list = mask_list.cuda()

            y_pred = model(sentence= word_list,mask= mask_list)
            score, pre, label = f1_score_evaluation(batch_masks= mask_list,
                                        batch_labels= tag_list,
                                        batch_prediction= y_pred)
            score_list.append(score)
            pre_list += pre
            label_list += label

    for i in range(len(label_list)):
        if pre_list[i] == label_list[i]:
            hit += 1
        n += 1
    #score_list
    print("average-f1-score: "+str(np.mean(score_list)))
    print("acc: ", hit/n)
    return np.mean(score_list)

"""""""""
=============== !!!!!!!!!!!!!! 模型訓練 !!!!!!!!!!!!!! ==============
"""""""""

import copy
model = BiLSTM_CRF(
    embedding_dim = EMBEDDING_DIM,
    hidden_dim = HIDDEN_SIZE,
    batch_size = BATCH_SIZE,
    embs_vec = embs_npy,
    num_layer = NUM_LAYERS,
    dropout = DROPOUT)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(use_gpu)
if use_gpu:
    model.cuda()


# optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_set = WordDataset(train_list, train_mask_list ,train_tag_list)
dev_set = WordDataset(dev_list, dev_mask_list ,dev_tag_list)
dev_loader = DataLoader(dev_set, batch_size= BATCH_SIZE, shuffle= True)

if use_gpu: #word_list, mask_list, tag_list       
    train_set.word_list.to(torch.device("cuda:0"))  # put data into GPU entirely
    train_set.mask_list.to(torch.device("cuda:0"))
    train_set.tag_list.to(torch.device("cuda:0"))
    dev_set.word_list.to(torch.device("cuda:0"))
    dev_set.mask_list.to(torch.device("cuda:0"))
    dev_set.tag_list.to(torch.device("cuda:0"))
     
best_f1 = 0
for epoch in range(5):
    train_loader = DataLoader(train_set, batch_size= BATCH_SIZE, shuffle= True)
    score_ = train(epoch, model, optimizer)
    if score_ > best_f1:
        best_model = copy.deepcopy(model.state_dict())
        best_f1 = score_
    
torch.save(best_model, "best_model_batch_4.pkl")

"""""""""
=============== !!!!!!!!!!!!!! 模型訓練 !!!!!!!!!!!!!! ==============
"""""""""



"""
test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

test_path = 'test-submit.txt'
test_list , _ , test_mask_list, test_num_word = load_data(test_path, train= False)
test_set = WordDataset(test_list ,test_mask_list, _ , test= True)
test_loader = DataLoader(test_set, batch_size= BATCH_SIZE)
pre_list, label_list, vac_list, score_list = [], [], [], []
model.eval()   #不啟用 BatchNormalization 和 Dropout，保證BN和dropout不發生變化
model.cpu()
use_gpu = False
for step, (word_list, mask_list) in enumerate(test_loader):
    with torch.no_grad(): 

        y_pred = model(sentence=word_list,mask=mask_list)
        pre, word = test_f1_score_evaluation(batch_masks=mask_list,
                                    batch_prediction=y_pred,
                                    batch_words= word_list)
        pre_list.append(pre)
        vac_list.append(word)
#score_list
print("test",len(pre_list)," ", len(vac_list))

for i in range(len(word_list)):
    print(vocab_npy[word_list[i]])

file_ = []
for i in range(len(pre_list)):
    for j in range(len(pre_list[i])):
        if vocab_npy[vac_list[i][j]] != "<STA>" and vocab_npy[vac_list[i][j]] != "<END>":
            ar = str(vocab_npy[vac_list[i][j]]) + " " + tag_dict_list[pre_list[i][j]]
            tmp3 = [ar]
            file_.append(tmp3)
        elif vocab_npy[vac_list[i][j]] == "<END>":
            tmp3 = []
            file_.append(tmp3)

pred_file = 'pre_drop_0.5_hid_1024_batch_4_text_change.txt'

with open(pred_file, "w+", encoding="utf-8") as file:
    for line in file_:
        file.write(" ".join(line))
        file.write("\n")

"""
dev!!!!!!!!!
"""
test_path = 'dev.txt'
test_list ,test_tag_list, test_mask_list, test_num_word = load_data(test_path, train= False)
test_set = WordDataset(test_list, test_mask_list, test_tag_list)
test_loader = DataLoader(test_set, batch_size= BATCH_SIZE)
pre_list, label_list, vac_list, score_list = [], [], [], []
model.eval()   #不啟用 BatchNormalization 和 Dropout，保證BN和dropout不發生變化
model.cpu()
use_gpu = False
for step, (word_list, tag_list, mask_list) in enumerate(dev_loader):
    with torch.no_grad(): 

        y_pred = model(sentence=word_list,mask=mask_list)
        score, pre, label, word = test_f1_score_evaluation(batch_masks=mask_list,
                                    batch_labels=tag_list,
                                    batch_prediction=y_pred,
                                    batch_words= word_list)
        score_list.append(score)
        pre_list.append(pre)
        label_list.append(label)
        vac_list.append(word)
#score_list
print("average-f1-score:"+str(np.mean(score_list)))
print(len(pre_list)," ", len(label_list))

file_ = []
for i in range(len(label_list)):
    for j in range(len(label_list[i])):
        if vocab_npy[vac_list[i][j]] != "<STA>" and vocab_npy[vac_list[i][j]] != "<END>":
            ar = str(vocab_npy[vac_list[i][j]]) + " " + tag_dict_list[label_list[i][j]] + " " + tag_dict_list[pre_list[i][j]]
            tmp3 = [ar]
            file_.append(tmp3)
        elif vocab_npy[vac_list[i][j]] == "<END>":
            tmp3 = []
            file_.append(tmp3)

pred_file = 'dev_pre_drop_0.5_hid_1024_batch_4_text_change.txt'

with open(pred_file, "w+", encoding="utf-8") as file:
    for line in file_:
        file.write(" ".join(line))
        file.write("\n")

print(test_list.shape)
hit = 0
n = 0
for i in range(len(label_list)):
    for j in range(len(label_list[i])):
        if pre_list[i][j] == label_list[i][j]:
            hit += 1
        n += 1
print("Acc: ", hit/n)
print("test_word_num: ", test_num_word)
print('hit: ', hit)
print('n: ', n)
