import torch
import numpy as np
from gensim.models import KeyedVectors
from torch.utils.data import Dataset
import pandas as pd
import os
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pickle
import random

import string


class ROCloader_train(Dataset):
    def __init__(self, directory,  prefix_length, suffix_length):

        self.prefix_length = prefix_length
        self.suffix_length = suffix_length
        self.storys = []


        w_2_c_path =os.path.join(directory, 'wiki-news-300d-1M.vec')
        # loading pretrained 300d word2vec
        if os.path.isfile(os.path.splitext(w_2_c_path)[0]+'.cache'):
            with open(os.path.splitext(w_2_c_path)[0] + '.cache', 'rb') as f:
                word2vec_wiki_300 = pickle.load(f)
        else:
            word2vec_wiki_300 = KeyedVectors.load_word2vec_format(w_2_c_path, binary=False)
            with open(os.path.splitext(w_2_c_path)[0]+'.cache', 'wb') as f:
                pickle.dump(word2vec_wiki_300, f, pickle.HIGHEST_PROTOCOL)


        file_name = 'ROCStories__spring2016 - ROCStories_spring2016.csv'

        file_path = os.path.join(directory, file_name)

        story_set = pd.read_csv(file_path)

        story_end = list(story_set['sentence5'])

        for rows in story_set.iterrows():

            pos_sample = {}
            neg_sample = {}

            prefix = [word for word in word_tokenize(" ".join(list(rows[1][2:6]))) if word not in string.punctuation]

            for index, token in enumerate(prefix):
                if token not in word2vec_wiki_300.vocab:
                    prefix[index] = np.zeros(300)
                else:
                    prefix[index] = word2vec_wiki_300[token]
                    # print(type(prefix[index]), prefix[index].shape)

            suffix_good = [word for word in word_tokenize(rows[1][6]) if word not in string.punctuation]

            for index, token in enumerate(suffix_good):
                if token not in word2vec_wiki_300.vocab:
                    suffix_good[index] = np.zeros(300)
                else:
                    suffix_good[index] = word2vec_wiki_300[token]

            #pop the correct end
            story_end_neg = story_end.pop(rows[0])
            #random choice from the rest
            suffix_bad = [word for word in word_tokenize(random.choice(story_end)) if word not in string.punctuation]
            #insert back the correct one
            story_end.insert(rows[0], story_end_neg)

            for index, token in enumerate(suffix_bad):
                if token not in word2vec_wiki_300.vocab:
                    suffix_bad[index] = np.zeros(300)
                else:
                    suffix_bad[index] = word2vec_wiki_300[token]


            pos_sample['prefix'] = prefix
            neg_sample['prefix'] = prefix

            pos_sample['suffix'] = suffix_good
            pos_sample['gt_class'] = 1

            neg_sample['suffix'] = suffix_bad
            neg_sample['gt_class'] = 0


            self.storys.append(pos_sample)
            self.storys.append(neg_sample)

    def __len__(self):
        return len(self.storys)

    def __getitem__(self, idx):
        prefix = np.array(self.storys[idx]['prefix'])

        prefix_pad = np.zeros((self.prefix_length, 300))
        prefix_pad[:prefix.shape[0], :] = prefix

        sufix = np.array(self.storys[idx]['suffix'])

        sufix_pad = np.zeros((self.suffix_length, 300))
        sufix_pad[:sufix.shape[0], :] = sufix

        label = np.array(self.storys[idx]['gt_class'])
        return torch.from_numpy(prefix_pad.transpose()), torch.from_numpy(sufix_pad.transpose()), torch.from_numpy(label)

        #for check the correctness of the loader
        # prefix = self.storys[idx]['prefix']
        # sufix = self.storys[idx]['suffix']
        # label = self.storys[idx]['gt_class']
        #
        # return prefix, sufix, label


if __name__ == '__main__':
    data = ROCloader_train('../dataset/',  100, 20)
    prifix, sufix, label = data[9]
    print(prifix, '\n', sufix, '\n', label)
