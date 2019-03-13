import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

class ROC_CNN(nn.Module):
    def __init__(self, input_channels):
        super(ROC_CNN).__init__()
        self.conv1_prefix = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv1_sufix = nn.Sequential(
            nn.Conv1d(input_channels=input_channels, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        pass



    def forward(self, *input):
        pass
