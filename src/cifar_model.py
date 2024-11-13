from torch import nn
import torch.nn.functional as F
import math

def make_layers_Cifar10(cfg, quant, batch_norm=False, conv=nn.Conv2d):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            use_quant = v[-1] != 'N'
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = conv(in_channels, filters, kernel_size=3, padding=1, bias=False)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU()]
            if quant!=None: layers += [quant()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)

class CNNCifar(nn.Module):
    def __init__(self, args,quant):
        self.args=args
        super(CNNCifar, self).__init__()
        self.linear = nn.Linear
        cfg = {
            9: ['64', '64', 'M', '128', '128', 'M', '256', '256', 'M'],
            11: ['64', 'M', '128', 'M', '256', '256', 'M', '512', '512', 'M', '512', '512', 'M'],
            13: ['64', '64', 'M', '128', '128', 'M', '256', '256', 'M', '512', '512', 'M', '512', '512', 'M'],
            16: ['64', '64', 'M', '128', '128', 'M', '256', '256', '256', 'M', '512', '512', '512', 'M', '512', '512', '512', 'M'],
        }
        self.conv = nn.Conv2d
        self.features = make_layers_Cifar10(cfg[16], quant, True, self.conv)
        self.classifier=None
        if quant!=None:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                self.linear(512 * 1 * 1, 4096),
                nn.ReLU(True),
                quant(),
                self.linear(4096, 4096),
                nn.ReLU(True),
                quant(),
                self.linear(4096, args.num_classes),
                nn.ReLU(True),
                quant(),
                nn.LogSoftmax(dim=1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                self.linear(512 * 1 * 1, 4096),
                nn.ReLU(True),
                self.linear(4096, 4096),
                nn.ReLU(True),
                self.linear(4096, args.num_classes),
                nn.ReLU(True),
                nn.LogSoftmax(dim=1)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.classifier(x)
        return x


class ResnetCifar18(nn.Module):
    def __init__(self, quant, quantx, in_channel, out_channel, strides):
        super(ResnetCifar18,self).__init__()
        self.block=None
        self.residual=nn.Sequential()
        self.quantx=quantx
        if quant==None:
            self.block=nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
            if strides!=1 or in_channel!=out_channel:
                self.residual=nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides,bias=False),
                    nn.BatchNorm2d(out_channel)
                )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                quant(),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                quant()
            )
            self.residual = nn.Sequential()
            if strides != 1 or in_channel != out_channel:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False),
                    nn.BatchNorm2d(out_channel),
                    quant()
                )
    def forward(self,x):
        out=self.block(x)
        out+=self.residual(x)
        out=F.relu(out)
        if self.quantx!=None:
            out=self.quantx(out)
        return out

class ResNet(nn.Module):
    def __init__(self, args, quant, quantx):
        super(ResNet,self).__init__()
        self.in_channel=64
        self.quantx=quantx
        self.conv1=None
        if quant==None:
            self.conv1=nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)# ,
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                quant(),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.layer1 = self.make_layer(quant, quantx, 64, 2, stride=1)
        self.layer2 = self.make_layer(quant, quantx, 128, 2, stride=2)
        self.layer3 = self.make_layer(quant, quantx, 256, 2, stride=2)
        self.layer4 = self.make_layer(quant, quantx, 512, 2, stride=2)
        self.fc=nn.Linear(512, args.num_classes)

    def make_layer(self, quant, quantx, channel, num_blocks, stride):
        strides=[stride] + [1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(ResnetCifar18(quant, quantx, self.in_channel, channel, stride))
            self.in_channel=channel
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        if self.quantx != None:
            out = self.quantx(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        if self.quantx != None:
            out = self.quantx(out)
        out = F.log_softmax(out,dim=1)
        return out
