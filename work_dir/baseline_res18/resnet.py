import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def adaptive_positional_encoding_4d(H, W):

    max_dist = torch.sqrt(torch.tensor(H**2 + W**2))

    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    i, j = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    dist = torch.sqrt((x.unsqueeze(2).unsqueeze(3) - j.unsqueeze(0).unsqueeze(1))**2 + \
                      (y.unsqueeze(2).unsqueeze(3) - i.unsqueeze(0).unsqueeze(1))**2) / max_dist

    # 动态调整衰减系数
    sigma = 0.6 * max_dist  
    slope = 0.5  

    positional_encoding = (1 - torch.clamp(slope * dist, 0, 1)) * torch.exp(-dist**2 / (2 * sigma**2))

    return positional_encoding



class Get_Correlation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels//16
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,1,1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,2,2), dilation=(1,2,2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,3,3), dilation=(1,3,3), groups=reduction_channel)

        

        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(6) / 2, requires_grad=True)
        self.weights3 = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):
        weights3_norm =  F.softmax(self.weights3, dim=0)
        x2 = self.down_conv2(x)
        positional_encode = adaptive_positional_encoding_4d(x.shape[3],x.shape[3]).unsqueeze(0).unsqueeze(0)
        device = torch.device("cuda:0")  # 选择第一个CUDA设备
        positional_encode = positional_encode.to(device) 
        torch.cuda.empty_cache()
        affinities = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2))  # repeat the last frame
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2))  # repeat the first frame 
        # affinities = affinities * positional_encode
        # affinities2 = affinities2 * positional_encode
        torch.cuda.empty_cache()
        cor11 =torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2), F.sigmoid(affinities)-0.5 )
        cor12 =torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2), F.sigmoid(affinities2)-0.5 )
        features1 = cor11 * self.weights2[0] + \
             cor12 * self.weights2[1] 
        
        # cor11,cor12,affinities,affinities2  = None,None,None ,None
        # torch.cuda.empty_cache()
        # affinities21 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,2:], x2[:,:,-2:]], 2))  # repeat the last frame 
        # #affinities21 = affinities21 #* positional_encode
        # affinities22 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,:2], x2[:,:,:-2]], 2))  # repeat the first frame  
        # #affinities22 = affinities22 * positional_encode
        # torch.cuda.empty_cache()
        # cor21 = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,2:], x2[:,:,-2:]], 2), F.sigmoid(affinities21)-0.5 )
        # cor22 = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,:2], x2[:,:,:-2]], 2), F.sigmoid(affinities22)-0.5 )
        # features2 = cor21 * self.weights2[2] + \
        #       cor22 * self.weights2[3]  
        # cor21,cor22,affinities22,affinities21  = None,None,None ,None        
        # features= weights3_norm[0] * features2 + weights3_norm[1] * features1
        # features2 = None
        # features1 = None
        # #torch.cuda.empty_cache()
        # affinities31 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,3:], x2[:,:,-3:]], 2))  # repeat the last frame 
        # affinities32 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,:3], x2[:,:,:-3]], 2))  # repeat the first frame  
        # #affinities31 = affinities31 * positional_encode
        # #affinities32 = affinities32 * positional_encode
        # torch.cuda.empty_cache()
        # cor31 =  torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,3:], x2[:,:,-3:]], 2), F.sigmoid(affinities31)-0.5 )
        # cor32 = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,:3], x2[:,:,:-3]], 2), F.sigmoid(affinities32)-0.5 )
        # features3 =cor31 * self.weights2[4] + \
        #          cor32* self.weights2[5]  
        # cor31,cor32,affinities31,affinities32  = None,None,None ,None
        # features = weights3_norm[2] * features3 + features
        # features3 = None
        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x)*self.weights[0] + self.spatial_aggregation2(x)*self.weights[1] \
                    + self.spatial_aggregation3(x)*self.weights[2] 
        aggregated_x = self.conv_back(aggregated_x)


        return features1 * (F.sigmoid(aggregated_x)-0.5)
        

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, dropout_rate=0.3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.corr1 = Get_Correlation(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr2 = Get_Correlation(self.inplanes)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr3 = Get_Correlation(self.inplanes)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, use_corr=True):
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x) 
        if use_corr:
            #cor1,identity1 = self.corr1(x)
            # x = x + cor1 * self.alpha[0] + identity1 * self.alpha[3]
            x = x + self.corr1(x) * self.alpha[0]
        x = self.layer3(x)
        if use_corr:
            #cor2,identity2 = self.corr2(x)
            #x = x + cor2 * self.alpha[1] + identity2 * self.alpha[4]
            x = x + self.corr2(x) * self.alpha[1]
        x = self.layer4(x)
        if use_corr:
            #cor3,identity3 = self.corr3(x)
            #x = x + cor3 * self.alpha[2] + identity3 * self.alpha[5]
            x = x + self.corr3(x) * self.alpha[2]
        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:]) #bt,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #bt,c
        x = self.dropout(x) 
        x = self.fc(x) #bt,c

        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model
def resnet34(**kwargs):
    """
    Constructs a ResNet-34 model.
    """

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    for name, param in checkpoint.items():
        if 'conv' in name or 'downsample.0.weight' in name:
            checkpoint[name] = param.unsqueeze(2)
  
    model.load_state_dict(checkpoint, strict=False)
    
    return model



def test():
    net = resnet34()
    device = torch.device("cuda:0")
    net.to(device)
    x = torch.randn(1,3,5,224,224)
    y = net(x.to(device))
    print(y.size())

#test()
