import torch 
import torch.nn as nn 


''' 
Residual block is a type of neural network layer that allows gradients
to flow more easily during backpropagation, facilitating the training of
deeper networks. The primary concept is to use a shortcut connection that
bypasses one or more layers, enabling the network to learn residual fn.  
rather than the original unreferenced functions.
'''


class ResidualBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,stride=1):
        super(ResidualBlock,self).__init__() 
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                        stride=stride,kernel_size=3,padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                        stride=stride,kernel_size=3,padding=1), 
            nn.BatchNorm2d(out_channels))

        ''' 
        This 1×1 convolution should adjust the dimensions of 
        residual to match the output of the convolutional layers. 
        '''
        
        if in_channels != out_channels or stride != 1 :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(out_channels)) 
            
        else:
            self.shortcut = nn.Identity() 

        
    #forward pass
    def forward(self,x):
        residual = x 
        out = self.conv1(x)
        out = self.conv2(out)  
        
        
        residual = self.shortcut(residual )
        out += residual 
        torch.relu(out)
        return out
        
        
        
class ResNet50():
    pass