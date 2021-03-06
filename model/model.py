import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision


class Conv1d (nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride,same_padding, bias=True, relu=True):
        super(Conv1d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv1d(input_channels,output_channels,kernel_size,stride,padding,bias=bias)
        self.relu = nn.ReLU() if relu else None

    def forward(self, input):
        x =self.conv(input)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ROC_CNN(nn.Module):
    def __init__(self, input_channels, training=True):
        super(ROC_CNN, self).__init__()
        self.training =training
        self.conv1_prefix = nn.Sequential(
            Conv1d(input_channels,256,3,1,same_padding=True,bias=True,relu=True),
            Conv1d(256, 256, 3, 1, same_padding=True, bias=True, relu=True),
            Conv1d(256, 256, 3, 1, same_padding=True, bias=True, relu=True),
        )

        self.conv2_prefix = nn.Sequential(
            Conv1d(256, 512, 3, 1, same_padding=True, bias=True, relu=True),
            Conv1d(512, 512, 3, 1, same_padding=True, bias=True, relu=True),
            Conv1d(512, 512, 3, 1, same_padding=True, bias=True, relu=True),
        )

        self.conv3_prefix = nn.Sequential(
            Conv1d(512, 512, 3, 1, same_padding=True, bias=True, relu=True),
            Conv1d(512, 512, 3, 1, same_padding=True, bias=True, relu=True),
            Conv1d(512, 512, 3, 1, same_padding=True, bias=True, relu=True),
        )

        self.conv1_sufix = nn.Sequential(
            Conv1d(input_channels,256,3,1,same_padding=True,bias=True,relu=True),
            Conv1d(256, 256, 3, 1, same_padding=True, bias=True, relu=True),
            Conv1d(256, 256, 3, 1, same_padding=True, bias=True, relu=True),
            )
        self.conv2_sufix = nn.Sequential(
            Conv1d(256, 512, 3, 1, same_padding=True, bias=True, relu=True),
            Conv1d(512, 512, 3, 1, same_padding=True, bias=True, relu=True),
            Conv1d(512, 512, 3, 1, same_padding=True, bias=True, relu=True),
        )


        self.linear_1 = nn.Linear(15360,4096,True)
        self.linear_2 = nn.Linear(4096, 2, True)



    def forward(self, input, label):
        batch_size = input.shape[0]
        conv1_prefix = F.max_pool1d(self.conv1_prefix(input),2)
        conv2_prefix = F.max_pool1d(self.conv2_prefix(conv1_prefix),2)
        # conv3_prefix = F.max_pool1d(self.conv3_prefix(conv2_prefix),2)

        #conv1_suffix = F.max_pool1d(self.conv1_sufix(suffix),2)
        # conv2_suffix = F.max_pool1d(self.conv2_sufix(conv1_suffix),2)

        # conv3_prefix_linear = conv3_prefix.view(batch_size, -1)
        # conv2_suffix_linear = conv2_suffix.view(batch_size, -1)

        conv2_prefix_linear = conv2_prefix.view(batch_size,-1)
        #conv2_suffix_linear = conv1_suffix.view(batch_size,-1)
        #fuse_ps = torch.cat((conv3_prefix_linear,conv2_suffix_linear),dim=1)

        linear_1 = F.relu(self.linear_1(conv2_prefix_linear))
        linear_2 = self.linear_2(linear_1)
        if self.training:

            return F.cross_entropy(linear_2,label)

        else:

            return F.softmax(linear_2)


