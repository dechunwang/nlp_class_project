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


class ROCloader(Dataset):
    def __init__(self, directory, mode, prefix_length, suffix_length):

        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.storys = []
        self.storys_raw =[]


        w_2_c_path =os.path.join(directory, 'wiki-news-300d-1M.vec')
        # loading pretrained 300d word2vec
        if os.path.isfile(os.path.splitext(w_2_c_path)[0]+'.cache'):
            with open(os.path.splitext(w_2_c_path)[0] + '.cache', 'rb') as f:
                word2vec_wiki_300 = pickle.load(f)
        else:
            word2vec_wiki_300 = KeyedVectors.load_word2vec_format(w_2_c_path, binary=False)
            with open(os.path.splitext(w_2_c_path)[0]+'.cache', 'wb') as f:
                pickle.dump(word2vec_wiki_300,f, pickle.HIGHEST_PROTOCOL)

        if mode == "test":
            file_name = 'cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
        elif mode == "val":
            file_name = 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'

        file_path = os.path.join(directory, file_name)

        story_set =pd.read_csv(file_path)

        for rows in story_set.iterrows():

            pos_sample ={}
            neg_sample ={}
            # set['prefix'] =
            # prefix = [word for word in word_tokenize(" ".join(list(rows[1][1:5]))) if word not in string.punctuation]
            # suffix_1 = [word for word in word_tokenize(rows[1][5]) if word not in string.punctuation]
            # suffix_2 = [word for word in word_tokenize(rows[1][6]) if word not in string.punctuation]

            prefix = [word for word in word_tokenize(" ".join(list(rows[1][1:5]))) if word not in string.punctuation]

            # for index, token in enumerate(prefix):
            #     if token not in word2vec_wiki_300.vocab:
            #         prefix[index] = np.zeros(300)
            #     else:
            #         prefix[index] = word2vec_wiki_300[token]
            #         # print(type(prefix[index]), prefix[index].shape)

            suffix_1 = [word for word in word_tokenize(rows[1][5]) if word not in string.punctuation]

            # for index, token in enumerate(suffix_1):
            #     if token not in word2vec_wiki_300.vocab:
            #         suffix_1[index] = np.zeros(300)
            #     else:
            #         suffix_1[index] = word2vec_wiki_300[token]

            suffix_2 = [word for word in word_tokenize(rows[1][6]) if word not in string.punctuation]


            # for index, token in enumerate(suffix_2):
            #     if token not in word2vec_wiki_300.vocab:
            #         suffix_2[index] = np.zeros(300)
            #     else:
            #         suffix_2[index] = word2vec_wiki_300[token]

            ending_class = rows[1][7]

            pos_sample['prefix'] = " ".join(list(rows[1][1:5]))
            neg_sample['prefix'] = " ".join(list(rows[1][1:5]))
            if ending_class == 1 :
                pos_sample['suffix'] = rows[1][5]
                pos_sample['gt_class'] = 1

                neg_sample['suffix'] = rows[1][6]
                neg_sample['gt_class'] = 0
            else:
                pos_sample['suffix'] = rows[1][6]
                pos_sample['gt_class'] = 1

                neg_sample['suffix'] = rows[1][5]
                neg_sample['gt_class'] = 0

            self.storys_raw.append(pos_sample)
            self.storys_raw.append(neg_sample)

            pos_sample=pos_sample.copy()
            neg_sample=neg_sample.copy()
            pos_sample['prefix'] = self.token_to_embed(prefix,word2vec_wiki_300)
            neg_sample['prefix'] = self.token_to_embed(prefix,word2vec_wiki_300)
            if ending_class == 1:
                pos_sample['suffix'] = self.token_to_embed(suffix_1,word2vec_wiki_300)
                pos_sample['gt_class'] = 1

                neg_sample['suffix'] = self.token_to_embed(suffix_2,word2vec_wiki_300)
                neg_sample['gt_class'] = 0
            else:
                pos_sample['suffix'] = self.token_to_embed(suffix_2,word2vec_wiki_300)
                pos_sample['gt_class'] = 1

                neg_sample['suffix'] = self.token_to_embed(suffix_1,word2vec_wiki_300)
                neg_sample['gt_class'] = 0

            self.storys.append(pos_sample)
            self.storys.append(neg_sample)



    def token_to_embed(self,tokens,word2vec_wiki_300):
        embeded=[]
        for token in (tokens):
            if token not in word2vec_wiki_300.vocab:
                embeded.append(np.zeros(300))
            else:
                embeded.append(word2vec_wiki_300[token])

        return embeded;

    def __len__(self):
        return len(self.storys)

    def __getitem__(self, idx):
        prefix = np.array( self.storys[idx]['prefix'])

        prefix_pad = np.zeros((self.prefix_length,300))
        prefix_pad[:prefix.shape[0],:]=prefix

        sufix =  np.array(self.storys[idx]['suffix'])

        sufix_pad = np.zeros((self.suffix_length, 300))
        sufix_pad[:sufix.shape[0], :] = sufix

        label = np.array(self.storys[idx]['gt_class'])
        return torch.from_numpy(prefix_pad.transpose()), torch.from_numpy(sufix_pad.transpose()), torch.from_numpy(label)


if __name__ == '__main__':
    data= ROCloader('../dataset/','val',100,20)
    prifix, sufix, label = data[8]
    print(prifix)
