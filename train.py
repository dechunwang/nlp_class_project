import torch
from util.ROCloader import *
from model.model import *


if __name__ == '__main__':
    epocs = 65
    #epoc_loss = []
    batch_size = 256
    #num_workers = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = ROC_CNN(300).to(device)


    dataset = ROCloader()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=4)


    #criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)



    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)
    for epoc in range (1, epocs+1):
        scheduler.step()
        batch_loss = []
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()

            prefix_input, sufix_input, target = batch
            prefix_input = prefix_input.to(device)
            sufix_input = sufix_input.to(device)
            target = target.to(device)

            loss = model(prefix_input, sufix_input, target)

            loss.backward()
            batch_loss.append(loss.item())
            optimizer.step()



        print(sum(batch_loss)/len(batch_loss))
