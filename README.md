### Implementation of Resnet in PyTorch

Implementation of "Deep Residual Learning for Image Recognition" from scratch in pure Pytorch.       

#### Key Components

The architecture begins with a convolutional layer that processes the input image and extracts features through filters, producing 64 feature maps. Following this, batch normalization is applied to the output of the convolutional layer, enhancing both the speed and stability of the training process. The ReLU activation function is then utilized to introduce non-linearity, enabling the model to learn complex patterns in the data.

One of the most important features of ResNet-50 is its use of identity blocks. These blocks allow the original input to bypass certain layers, which helps maintain the flow of information through the network and makes training more manageable. Additionally, pooling layers are included to reduce the size of the feature maps, concentrating on the most significant features and making computations more efficient.

At the end of the architecture, a fully connected layer takes the final output from the preceding layers and generates classification scores for each class in the dataset.

ResNet-50 stands out due to its effective residual learning approach, which simplifies the task of training very deep networks. The inclusion of skip connections allows gradients to flow easily during training, thereby preventing issues like vanishing gradients. As a result, ResNet-50 often outperforms many traditional models, making it a favored choice for various image classification tasks.

#### Why ResNet

- **Residual Learning**: The main idea of ResNet is that it allows the model to learn the difference between the output and the input (the residual), which makes it easier to train very deep networks.
- **Skip Connections**: These connections help gradients flow through the network without vanishing, allowing for better training of deeper models.
- **Performance**: ResNet-50 has shown to perform better on various image classification tasks compared to traditional networks of similar depth due to its innovative architecture.


#### Advantages of this Architecture

- **Addressing Vanishing Gradients**:
      Traditional deep networks often face the problem of vanishing gradients, where the gradients become very small as they propagate back through many layers, making training difficult. ResNet’s skip connections help preserve the gradient, allowing for more effective training of deeper networks.

- **Better Performance**:
  ResNet architectures, including ResNet-50, have consistently outperformed many other deep learning architectures in image classification tasks, achieving state-of-the-art results in competitions like ImageNet.

#### References

You can access the paper [here](https://arxiv.org/pdf/1512.03385).                  
This project is based on the following paper: 
He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep Residual Learning for Image Recognition." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778. 2016.
