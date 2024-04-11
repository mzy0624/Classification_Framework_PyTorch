import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class BlockUnit(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=0, relu=False):
        super(BlockUnit, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, planes, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        self.norm = nn.BatchNorm2d(planes)
        self.relu = F.relu if relu else lambda x : x
    
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class Block(nn.Module):
    def __init__(self, residual_func, expansion, in_planes, planes, stride=1):
        super(Block, self).__init__()
        self.residual = residual_func
        self.shortcut = BlockUnit(
            in_planes,
            planes * expansion,
            kernel_size=1,
            stride=stride
        ) if stride != 1 or in_planes != planes * expansion else lambda x : x

    def forward(self, x):
        return F.relu(self.residual(x) + self.shortcut(x))
        
class BasicBlock(Block):
    expansion = 1
    nlayer = 2
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__(
            nn.Sequential(
                BlockUnit(in_planes, planes, stride=stride, padding=1, relu=True),
                BlockUnit(planes, planes * self.expansion, padding=1)
            ),
            self.expansion, in_planes, planes, stride
        )

class BottleNeck(Block):
    expansion = 4
    nlayer = 3
    def __init__(self, in_planes, planes, stride=1):
        super(BottleNeck, self).__init__(
            nn.Sequential(
                BlockUnit(in_planes, planes, kernel_size=1, relu=True),
                BlockUnit(planes, planes, stride=stride, padding=1, relu=True),
                BlockUnit(planes, planes * self.expansion, kernel_size=1)
            ),
            self.expansion, in_planes, planes, stride
        )

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv = BlockUnit(3, 64, padding=1, relu=True)
        self.layers = nn.Sequential(
            self._make_layer(block,  64, num_blocks[0]),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2)
        )
        self.linear  = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(4)
        self.nlayer  = 2 + sum(num_blocks) * block.nlayer
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # init.xavier_uniform_(module.weight)
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
    def __str__(self):
        return f'ResNet{self.nlayer}'


class ResNet18(ResNet):
    def __init__(self, num_classes):
        super( ResNet18,  self).__init__(BasicBlock, [2, 2,  2, 2], num_classes)
        
class ResNet34(ResNet):
    def __init__(self, num_classes):
        super( ResNet34,  self).__init__(BasicBlock, [3, 4,  6, 3], num_classes)

class ResNet50(ResNet):
    def __init__(self, num_classes):
        super( ResNet50,  self).__init__(BottleNeck, [3, 4,  6, 3], num_classes)

class ResNet101(ResNet):
    def __init__(self, num_classes):
        super(ResNet101,  self).__init__(BottleNeck, [3, 4, 23, 3], num_classes)

class ResNet152(ResNet):
    def __init__(self, num_classes):
        super(ResNet152,  self).__init__(BottleNeck, [3, 8, 36, 3], num_classes)