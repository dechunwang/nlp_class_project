import torch
from util.ROCloader import *
from util.ROC_train_loader import *
from util.bert_test_loader import *
from util.bert_emb_loader import *
from model.model import *


if __name__ == '__main__':

    PATH = 'check_point/40check_point.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = bert_test_loader('dataset/', prefix_length=100, suffix_length=20)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    th = 0.6

    model = ROC_CNN(768, training=False).to(device)
    model = nn.DataParallel(model)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    correct = 0

    for i, batch in enumerate(dataloader):
        text, target = batch

        text = text.to(device).float()
        target = target.to(device).long()
        #print(text.shape, target)
        output = model(text, target)

        predit = (output[0, 1] > th).sum()

        print(predit, target)

        if target == predit:

            correct += 1

    print(correct/len(dataloader))
