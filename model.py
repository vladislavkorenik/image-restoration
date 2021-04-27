import torch 
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_size, use_activation=True):
        
        super(ConvBlock, self).__init__()
        self.use_activation = use_activation
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        pred = self.bn(self.conv(x))
        return self.activation(pred) if self.use_activation else pred


class ResultBlock(nn.Module):
    
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResultBlock, self).__init__()
        self.block1 = ConvBlock(input_channels, output_channels, kernel_size)
        self.block2 = ConvBlock(input_channels, output_channels, kernel_size, use_activation=False)

    def forward(self, x):
        return x + self.block2(self.block1(x))
    

class NetModel(nn.Module):

    def __init__(self, input_channels, output_channels, result_layers=16):
        super(NetModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        _result = [ResultBlock(output_channels, output_channels, 3) for i in range(result_layers)]
        self.result = nn.Sequential(*_result)

        self.conv2 = ConvBlock(output_channels, output_channels, 3, use_activation=False)
        self.conv3 = nn.Conv2d(output_channels, input_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, input):
        pred1 = self.activation(self.conv1(input))
        pred2 = self.conv2(self.result(pred1))
        pred = self.conv3(torch.add(pred1, pred2))
        return pred
