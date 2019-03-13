import torch
from ""


epocs = 50

epoc_loss = []


for epoc in range (1, epocs+1):
    batch_loss = []
    for