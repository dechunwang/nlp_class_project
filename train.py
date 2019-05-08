import torch
from util.ROCloader import *
from util.ROC_train_loader import *
from util.bert_emb_loader import *
from model.model import *


if __name__ == '__main__':
    epocs = 40
    #epoc_loss = []
    batch_size = 4000
    #num_workers = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = ROC_CNN(768, training=True).to(device)
    model = nn.DataParallel(model)

    dataset = bert_loader('dataset/', 100, 20)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print('Total: {}'.format(len(dataset)))

    #criterion = torch.nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr=0.01)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1, last_epoch=-1)
    model.train()
    for epoc in range(1, epocs+1):
        scheduler.step()
        batch_loss = []
        for i, batch in enumerate(dataloader):

            optimizer.zero_grad()

            text, target = batch
            text = text.to(device).float()
            #sufix_input = sufix_input.to(device).float()
            target = target.to(device).long()
            loss = model(text, target)
            loss = loss.sum()
            print("loss:", loss.item())
            loss.backward()
            batch_loss.append(loss.item())
            optimizer.step()



        print('epoic:{}  loss:{}'.format(epoc,np.array(batch_loss).mean()))


        if epoc%10 ==0:
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'check_point/%scheck_point.pth' %epoc)