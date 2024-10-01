import torch 
import torch.nn as nn 
import torch.nn.functional as F


''' 
Residual block is a type of neural network layer that allows gradients
to flow more easily during backpropagation, facilitating the training of
deeper networks. The primary concept is to use a shortcut connection that
bypasses one or more layers, enabling the network to learn residual fn.  
rather than the original unreferenced functions.
'''

# Residual Network
class ResidualBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, stride=1):
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
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels)) 
            
        else:
            self.shortcut = nn.Identity() 

        
    #forward pass
    def forward(self,x):
        residual = x 
        out = self.conv1(x)
        out = self.conv2(out)  
        
        
        residual = self.shortcut(residual)
        out += residual 
        out = F.relu(out)
        return out
        
        
# Main Function    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet,self).__init__() 
        
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,
                               padding=3,stride=2)
    
        
        self.bn1 = nn.BatchNorm2d(64) 
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
         
        
            
    def _make_layer(self, block, num_blocks, planes, stride=1):
        
        # Create a list of strides
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            
            # Create the block
            layers.append(block(self.in_planes, planes, stride))  
            
             # Update in_planes
            self.in_planes = planes * block.expansion 
        return nn.Sequential(*layers) 
    
    def forward(self,x):
      
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
         # Reduce to a single value for each feature map
        out = F.avg_pool2d(out)
        
         # Flatten to prepare for the fully connected layer
        out = out.view(out.size(0), -1) 
        
        # Final classification scores
        out = self.linear(out)
        
        # Return the output! Yay!
        return out
    
    
