### Implementation of Resnet in PyTorch

Implementation of "Deep Residual Learning for Image Recognition" from scratch in pure Pytorch.       

#### Architecture

**ResNet-50 Overview**
## ResNet-50 Architecture

| **Layer Type**        | **Output Size**      | **Parameters** | **Description**                                           |
|-----------------------|----------------------|----------------|-----------------------------------------------------------|
| **Input**             | 3 x 224 x 224        | -              | Input image with 3 color channels (RGB) and size 224x224. |
| **Conv1**             | 64 x 112 x 112       | 9,472          | 7x7 convolution, stride 2, padding 3, 64 filters.         |
| **BatchNorm**         | 64 x 112 x 112       | 128            | Batch normalization for Conv1 output.                     |
| **ReLU**              | 64 x 112 x 112       | -              | Activation function.                                       |
| **Max Pooling**       | 64 x 56 x 56         | -              | 3x3 max pooling, stride 2, padding 1.                    |

### Residual Layers

| **Layer Block**       | **Number of Blocks** | **Output Size**      | **Parameters per Block** | **Description**                                         |
|-----------------------|----------------------|----------------------|--------------------------|---------------------------------------------------------|
| **Layer 1**           | 3                    | 64 x 56 x 56         | -                        | Residual blocks with 64 filters, stride 1.            |
| **Layer 2**           | 4                    | 128 x 28 x 28        | -                        | Residual blocks with 128 filters, stride 2.           |
| **Layer 3**           | 6                    | 256 x 14 x 14        | -                        | Residual blocks with 256 filters, stride 2.           |
| **Layer 4**           | 3                    | 512 x 7 x 7          | -                        | Residual blocks with 512 filters, stride 2.           |

### Final Layers

| **Layer Type**        | **Output Size**      | **Parameters** | **Description**                                           |
|-----------------------|----------------------|----------------|-----------------------------------------------------------|
| **Adaptive Average Pooling** | 512 x 1 x 1      | -              | Reduces feature map to a single value per channel.       |
| **Flatten**           | 512                  | -              | Converts 3D feature maps to 1D vector.                   |
| **Linear (Fully Connected)** | num_classes (10) | 5,130          | Outputs class scores (num_classes = 10).                 |

### Total Parameters

- **Total Parameters**: Approximately 23 million (23M)


#### Advantages of this Architecture

  Addressing Vanishing Gradients:
      Traditional deep networks often face the problem of vanishing gradients, where the gradients become very small as they propagate back through many layers, making training difficult. ResNet’s skip connections help preserve the gradient, allowing for more effective training of deeper networks.

  Better Performance:
