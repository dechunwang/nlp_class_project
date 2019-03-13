import torch
from util.ROCloader import *
from model.model import *


if __name__ == '__main__':

    PATH = 'check_point/check_point.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ROCloader('dataset/',mode='test', prefix_length=100, suffix_length=20)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    th = 0.6

    model = ROC_CNN(300).to(device)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint[0])

    correct = 0

    for i, batch in dataloader:

        prefix, sufix, target = batch

        output = model(prefix, sufix, target)

        if (output[:,1] > th).sum() == 1:
            correct+=1

    print(correct/len(dataloader))
