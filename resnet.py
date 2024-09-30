import torch 
import torch.nn as nn 


#creating ResidualBlock 
class ResidualBlock(nn.Module):
    def __init__(self,in_channels:int,out_channels:int,stride=1):
        super(ResidualBlock,self).__init__() 
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=3,padding=1), 
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,stride=stride,kernel_size=3,padding=1), 
<<<<<<< HEAD
            nn.BatchNorm2d(out_channels))

        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
            )
                    
    
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out) 
        
        
        pass 
=======
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        
    
    def forward(x):
        pass 
        
        
>>>>>>> 9c8d00a95ab0ab5369822750837b37ded4251a2f
