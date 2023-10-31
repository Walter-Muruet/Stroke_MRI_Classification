import torch
import torch.nn as nn

def convolution_layer(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, pool_size:int = 2,) -> torch.nn.Module:
    """
    Abstracts the logic to create a set of layers comprised of convolution, activation, batch normalisation and pooling.
    Params:
        in_channels: An integer passed to torch.nn.Conv2d. Number of input channels to the convolution layer.
        out_channels: An integer passed to torch.nn.Conv2d and nn.NatchNorm2d. Number of output channels to the convolution layer and Number of features to the Batch Normalisation layer.
        kernel_size: An interger passed to torch.nn.Conv2d. Size of the kernel (filter). The size of the kernel is kernel_size x kernel_size.
        stride: An integer passed to torch.nn.Conv2d. Size of the stride for the kernel.
        pool_size: An integer passed to torch.nn.MaxPool2d. Kernel size for the Max Pool 2D layer. The size of the kernel is pool_size x pool_size

    Returns:
        A torch.nn.Module comprised of the aforementioned set of layers
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels), #Alternatively, batch normalisation can be skipped if the number of parameters is reduced. This can be achieved by reducing the number of layers, increasing stride size, increasing pooling kernel size or downsizing the input image
        nn.MaxPool2d(pool_size)
    )

##############################################################

def convolution_layer2conv(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, pool_size:int = 2,) -> torch.nn.Module:
    """
    Abstracts the logic to create a set of layers comprised of (convolution, activation) x 2, batch normalisation and pooling.
    Params:
        in_channels: An integer passed to torch.nn.Conv2d. Number of input channels to the convolution layer.
        out_channels: An integer passed to torch.nn.Conv2d and nn.NatchNorm2d. Number of output channels to the convolution layer and Number of features to the Batch Normalisation layer.
        kernel_size: An interger passed to torch.nn.Conv2d. Size of the kernel (filter). The size of the kernel is kernel_size x kernel_size.
        stride: An integer passed to torch.nn.Conv2d. Size of the stride for the kernel.
        pool_size: An integer passed to torch.nn.MaxPool2d. Kernel size for the Max Pool 2D layer. The size of the kernel is pool_size x pool_size

    Returns:
        A torch.nn.Module comprised of the aforementioned set of layers
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride),
        nn.ReLU(),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels), #Alternatively, batch normalisation can be skipped if the number of parameters is reduced. This can be achieved by reducing the number of layers, increasing stride size, increasing pooling kernel size or downsizing the input image
        nn.MaxPool2d(pool_size)
    )