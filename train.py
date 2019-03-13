import torch
from util.ROCloader import *
from model.model import *


if __name__ == '__main__':
    epocs = 200
    #epoc_loss = []
    batch_size = 512
    #num_workers = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = ROC_CNN(300).to(device)


    dataset = ROCloader('dataset/','val',100,20)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=4)

    print('Total: {}'.format(len(dataset)))

    #criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr=0.01)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1, last_epoch=-1)
    model.train()
    for epoc in range (1, epocs+1):
        scheduler.step()
        batch_loss = []
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()

            prefix_input, sufix_input, target = batch
            prefix_input = prefix_input.to(device).float()
            sufix_input = sufix_input.to(device).float()
            target = target.to(device).long()

            loss = model(prefix_input, sufix_input, target)

            loss.backward()
            batch_loss.append(loss.item())
            optimizer.step()



        print('epoic:{}  loss:{}'.format(epoc,np.array(batch_loss).mean()))

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'check_point/check_point.pth')