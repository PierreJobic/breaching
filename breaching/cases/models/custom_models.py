import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class custom_McMahan_CNN(nn.Module):
    """Convolutional Neural Network architecture as described in McMahan 2017
    paper :
    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)

    Expected input_size: [N,C,28,28]
    """

    def __init__(self, num_classes=10, is_gray=False) -> None:
        super().__init__()
        if is_gray:
            self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
            self.input_shape = torch.Size([1, 28, 28])
        else:
            self.conv1 = nn.Conv2d(3, 32, 5, padding=1)
            self.input_shape = torch.Size([3, 28, 28])
        self.output_shape = torch.Size([num_classes])
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """expect input of size [N, 1/3, 28, 28]."""
        output_tensor = F.relu(self.conv1(x))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = nn.Flatten()(output_tensor)
        output_tensor = F.relu(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor


class custom_McMahan_32_32(nn.Module):
    """Modified CNN architecture to allow 32x32 images

    Expected input_size: [N,C,32,32]
    """

    def __init__(self, num_classes=10, is_gray=False) -> None:
        super().__init__()
        if is_gray:
            self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
            self.input_shape = torch.Size([1, 28, 28])
        else:
            self.conv1 = nn.Conv2d(3, 32, 5, padding=1)
            self.input_shape = torch.Size([3, 28, 28])
        self.output_shape = torch.Size([num_classes])
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """expect input of size [N, 1/3, 28, 28]."""
        # x -> conv1 -> ReLu -> Pool -> conv -> Relu -> Pool -> fc1 -> Relu -> fc2 -> loss
        # conv1.weight -> conv1.bias -> conv2.weight -> conv2.bias -> fc1.weight -> fc2.bias -> fc2.weight -> fc2.bias (all gradients)
        output_tensor = F.relu(self.conv1(x))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = nn.Flatten()(output_tensor)
        output_tensor = F.relu(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor  # (batch_size, num_classes)


class custom_batchnorm_McMahan_32_32(nn.Module):
    """Modified CNN architecture to allow 32x32 images

    Expected input_size: [N,C,32,32]
    """

    def __init__(self, num_classes=10, is_gray=False) -> None:
        super().__init__()
        if is_gray:
            self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
            self.input_shape = torch.Size([1, 28, 28])
        else:
            self.conv1 = nn.Conv2d(3, 32, 5, padding=1)
            self.input_shape = torch.Size([3, 28, 28])
        self.bn1 = nn.BatchNorm2d(32)
        self.output_shape = torch.Size([num_classes])
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """expect input of size [N, 1/3, 28, 28]."""
        # x -> conv1 -> ReLu -> Pool -> conv -> Relu -> Pool -> fc1 -> Relu -> fc2 -> loss
        # conv1.weight -> conv1.bias -> conv2.weight -> conv2.bias -> fc1.weight -> fc2.bias -> fc2.weight -> fc2.bias (all gradients)
        output_tensor = F.relu(self.bn1(self.conv1(x)))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.bn2(self.conv2(output_tensor)))
        output_tensor = self.pool(output_tensor)
        output_tensor = nn.Flatten()(output_tensor)
        output_tensor = F.relu(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor  # (batch_size, num_classes)


class custom_deeper_McMahan_CNN(nn.Module):
    """Convolutional Neural Network architecture as described in McMahan 2017
    paper :
    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)

    Expected input_size: [N,C,28,28]
    """

    def __init__(self, num_classes=10, is_gray=False) -> None:
        super().__init__()
        if is_gray:
            self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
            self.input_shape = torch.Size([1, 224, 224])
        else:
            self.conv1 = nn.Conv2d(3, 32, 5, padding=1)
            self.input_shape = torch.Size([3, 224, 224])
        self.output_shape = torch.Size([num_classes])
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, 5, padding=1)
        # self.conv4 = nn.Conv2d(128, 256, 5, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(200704, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """expect input of size [N, 1/3, 28, 28]."""
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        # output_tensor = F.sigmoid(self.conv3(output_tensor))
        # output_tensor = self.pool(output_tensor)
        # output_tensor = F.sigmoid(self.conv4(output_tensor))
        # output_tensor = self.pool(output_tensor)
        output_tensor = nn.Flatten()(output_tensor)
        output_tensor = F.sigmoid(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor


class custom_LeNet_smooth(nn.Module):
    """modified to handle EMNIST Balanced Images."""

    def __init__(self, num_classes=10, is_gray=True):
        super().__init__()
        act = nn.Sigmoid
        if is_gray:
            conv1 = nn.Conv2d(1, 12, kernel_size=5, padding=5 // 2, stride=2)
            self.input_shape = torch.Size([1, 28, 28])
        else:
            conv1 = nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2)
            self.input_shape = torch.Size([3, 28, 28])
        self.output_shape = torch.Size([num_classes])
        self.body = nn.Sequential(
            conv1,
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
            nn.Flatten(),
            nn.Linear(12 * 7 * 7, 100),
            act(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        out = self.body(x)
        return out


class custom_smooth_AlexNet(nn.Module):
    def __init__(self, num_classes=1000, is_gray=False):
        self.input_shape = torch.Size([3, 224, 224])
        self.output_shape = torch.Size([num_classes])
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.Sigmoid(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


class custom_LeNet_fromDLG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        act = nn.Sigmoid
        self.input_shape = torch.Size([3, 32, 32])
        self.output_shape = torch.Size([num_classes])
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(12 * 8 * 8, num_classes))

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = torch.sigmoid(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = torch.sigmoid(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, is_gray=False):
        """
        activation: sigmoid
        BatchNorm2d: True
        """
        super(ResNet, self).__init__()
        self.in_planes = 64
        if is_gray:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.output_shape = torch.Size([num_classes])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class custom_ResNet18(ResNet):
    """
    torchsummary.summary(ResNet18(num_classes=10), input_size=(3,28,28))
    ================================================================
    Total params: 11,173,962
    Trainable params: 11,173,962
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 8.79
    Params size (MB): 42.63
    Estimated Total Size (MB): 51.42
    ----------------------------------------------------------------
    """

    def __init__(self, num_classes=10, is_gray=False):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes, is_gray)
        if is_gray:
            self.input_shape = torch.Size([1, 28, 28])
        else:
            self.input_shape = torch.Size([3, 28, 28])


class custom_ResNet34(ResNet):
    def __init__(self, num_classes=10, is_gray=False):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes, is_gray)


class custom_ResNet50(ResNet):
    def __init__(self, num_classes=10, is_gray=False):
        super().__init__(Bottleneck, [3, 4, 6, 3], num_classes, is_gray)


class custom_ResNet101(ResNet):
    def __init__(self, num_classes=10, is_gray=False):
        super().__init__(Bottleneck, [3, 4, 23, 3], num_classes, is_gray)


class custom_ResNet152(ResNet):
    def __init__(self, num_classes=10, is_gray=False):
        super().__init__(Bottleneck, [3, 8, 36, 3], num_classes, is_gray)


class custom_AlexNet(nn.Module):
    def __init__(self, num_classes=10, *args, **kwargs):
        super().__init__()
        self.net = models.AlexNet(num_classes=num_classes)
        self.input_shape = torch.Size([3, 224, 224])
        self.output_shape = torch.Size([num_classes])

    def forward(self, x):
        return self.net(x)
