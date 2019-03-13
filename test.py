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
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct = 0

    for i in range(len(dataset)):

        prefix, sufix, target = dataset[i][0].to(device).float(),dataset[i][1].to(device).float(),dataset[i][2].to(device)

        output = model(prefix.unsqueeze(0), sufix.unsqueeze(0), target)

        predit=(output[0,1] > th).sum()

        print("{} \n {} \n we predict: {}  gt: {}".format(
            dataset.storys_raw[i]['prefix'],dataset.storys_raw[i]['suffix'],predit, target))

        if target==predit:

            correct += 1

    print(correct/len(dataloader))
