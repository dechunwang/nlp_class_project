
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import nltk


class ROCloader(Dataset):
    def __init__(self, directory, mode):
        self.storys = []


        if mode == "test":
            file_name ='cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
        elif mode == "val":
            file_name = 'cloze_test_val__spring2016 - cloze_test_ALL_val.csv'

        file_path = os.path.join(directory,file_name)

        story_set =pd.read_csv(file_path)
        sentence_column = story_set.axes[1]
        for rows in story_set.iterrows():

            set = {}





    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    data= ROCloader('../dataset/','train')
