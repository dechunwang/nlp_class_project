import torch
import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pickle
import string
import random
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import time

class bert_test_loader:
    def __init__(self, directory, prefix_length, suffix_length):
        self.prefix_length = prefix_length
        self.suffix_length = suffix_length



        file_name = 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'

        file_path = os.path.join(directory, file_name)

        story_set = pd.read_csv(file_path)


        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        self.bert_model.to('cuda')

        self.storys_raw = []

        if os.path.isfile('storyset_test_raw.cache'):
            with open('storyset_test_raw.cache', 'rb') as f:
                self.storys_raw = pickle.load(f)

        else:

            for rows in story_set.iterrows():
                start = time.time()
                pos_sample = {}
                neg_sample = {}

                prefix = [word for word in self.tokenizer.tokenize(" ".join(list(rows[1][1:5]))) if word not in string.punctuation]


                if rows[1][7] == '1':
                    suffix_good = [word for word in self.tokenizer.tokenize(rows[1][5]) if word not in string.punctuation]
                    suffix_bad = [word for word in self.tokenizer.tokenize(rows[1][6]) if word not in string.punctuation]
                else:
                    suffix_good = [word for word in self.tokenizer.tokenize(rows[1][6]) if word not in string.punctuation]
                    suffix_bad = [word for word in self.tokenizer.tokenize(rows[1][5]) if word not in string.punctuation]
                #pop the correct end

                pos_sentence = [prefix, suffix_good]
                pos_sample['text'] = pos_sentence
                pos_sample['gt_class'] = 1

                neg_sentence = [prefix, suffix_bad]
                neg_sample['text'] = neg_sentence
                neg_sample['gt_class'] = 0


                self.storys_raw.append(pos_sample)
                self.storys_raw.append(neg_sample)

                end = time.time()
                print(end-start)

            with open('storyset_test_raw.cache', 'wb') as g:
                pickle.dump(self.storys_raw, g, pickle.HIGHEST_PROTOCOL)

    def zerolistmaker(self, n):
        listofzeros = [0] * n
        return listofzeros

    def onelistmaker(self,n):
        listofones = [1] * n
        return listofones


    def sent_to_emb(self, bert_model, sentence):

        sentence1 = sentence[0]+sentence[1]
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(sentence1)
        indexed_tokens = torch.tensor([indexed_tokens]).to('cuda')
        #print(len(sentence[0]))
        prefix = self.zerolistmaker(len(sentence[0]))
        suffix = self.onelistmaker(len(sentence[1]))
        segments_ids = prefix + suffix
        segments_ids = torch.tensor([segments_ids]).to('cuda')
        indexed_tokens = indexed_tokens
        segments_ids = segments_ids
        with torch.no_grad():
            sent_emb, _ = bert_model(indexed_tokens, segments_ids)
        sent_emb = torch.cat(sent_emb)
        sent_emb = torch.max(sent_emb, 0)
        #print(sent_emb.type())
        return sent_emb[0].cpu()
    def __getitem__(self, idx):
        text = self.storys_raw[idx]['text']
        sent_emb = self.sent_to_emb(self.bert_model, self.storys_raw[idx]['text'])
        #sent_emb = np.array(self.storys[idx]['text'])
        sent_emb_pad =torch.tensor(np.zeros((self.prefix_length+self.suffix_length, 768)))
        sent_emb_pad[:sent_emb.shape[0], :] = sent_emb
        label = np.array(self.storys_raw[idx]['gt_class'])
        return sent_emb_pad.transpose(0, 1), torch.from_numpy(label)


    def __len__(self):
        return len(self.storys_raw)






if __name__ == '__main__':
    data = bert_test_loader('../dataset/',  100, 20)
    text, label = data[0]
    print(text.shape, label)