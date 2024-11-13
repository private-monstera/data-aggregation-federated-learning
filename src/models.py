from torch import nn
import math

def make_layers_Mnist(cfg, quant, batch_norm=False, conv=nn.Conv2d):
    layers = list()
    in_channels = 1
    n = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            use_quant = v[-1] != 'N'
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = conv(in_channels, filters, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(filters), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            if quant!=None: layers += [quant()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)

class CNNMnist(nn.Module):
    def __init__(self, args,quant):
        self.args=args
        super(CNNMnist, self).__init__()
        self.linear = nn.Linear
        cfg = {
            16: ['16', 'M', '32', 'M']
        }
        self.conv = nn.Conv2d
        self.features = make_layers_Mnist(cfg[16], quant, True, self.conv)
        self.classifier=None
        if quant!=None:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                self.linear(7*7*32, 512),
                nn.ReLU(True),
                quant(),
                self.linear(512, 10),
                nn.ReLU(True),
                quant(),
                nn.LogSoftmax(dim=1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                self.linear(7 * 7 * 32, 512),
                nn.ReLU(True),
                self.linear(512, 10),
                nn.ReLU(True),
                nn.LogSoftmax(dim=1)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x