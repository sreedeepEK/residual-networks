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

class Block(nn.Module):
    expansion = 4  # Class attribute for expansion

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * Block.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * Block.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x += identity
        return self.relu(x)

        
# Main Function    
class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels=3, num_classes=10):
        super(ResNet,self).__init__() 
        
        self.in_planes = 64
       
        
        self.conv1 = nn.Conv2d(in_channels=img_channels,out_channels=64,kernel_size=7, padding=3, stride=2)
    
        self.bn1 = nn.BatchNorm2d(64) 
        self.relu = nn.ReLU() 
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
           
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)
         
        
            
    def _make_layer(self, block, planes, num_blocks, stride=1):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_planes != planes * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers.append(block(self.in_planes, planes, identity_downsample, stride))
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
         # Reduce to a single value for each feature map
        out = self.avg_pool(out)
        
         # Flatten to prepare for the fully connected layer
        out = out.view(out.size(0), -1) 
        
        # Final classification scores
        out = self.linear(out)
        
        # Return the output! Yay!
        return out
    
def resnet50(img_channels=3, num_classes=1000):
    return ResNet(Block,[3,4,6,3],img_channels,num_classes)


# def main() -> None:
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     net = resnet50(img_channels=3, num_classes=10).to(device)
#     x = torch.randn(2, 3, 224, 224, device=device)

#     y: torch.Tensor = net(x)
#     print(f'{y.size() = }')

# if __name__ == '__main__':
#     main()