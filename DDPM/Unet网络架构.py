import torch
from torch import nn
import torch.nn.functional as F
"""
Unet的整体结构：输入层，左部分四层，右部分四层，输出层
"""
"""
Unet卷积层主体部分，实际上就是两次卷积
"""
class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv,self).__init__()
        """
        记住输出大小的计算公式(输入大小-卷积核大小+2*填充）/(步长)+1,一般是向下取整数
        下面对应的其实是图片中每一行的过程,在这里注意：按照文献中的表示padding=0，这里为了保持图像大小不变设置padding=1
        """
        self.double_conv=nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
                                nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return  self.double_conv(x)

"""
定义左部分分层编码，做了一次最大池化操作和一次基础性的两次卷积操作
"""
class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv=nn.Sequential(nn.MaxPool2d(2),
                                        DoubleConv(in_channels,out_channels))

    def forward(self,x):
        return self.maxpool_conv(x)


"""
右部分分层编码
"""
class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super().__init__()
        #下面的实际上是两种不同的上采样方法，插值法比较快不需要额外参数，逆卷积化需要训练参数但效果较好
        if bilinear:
            self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        else:
            self.up=nn.ConvTranspose2d(in_channels//2,in_channels//2,kernel_size=2,stride=2)
        self.conv=DoubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1=self.up(x1)  #实际上是做了一次上采样的操作
        #(batch_size,channel,height,width)实际上是 (批量数，通道数，行数，列数)
        diffY=torch.tensor(x2.size()[2]-x1.size()[2])  #算x1,x2行差
        diffX=torch.tensor(x2.size()[3]-x1.size()[3]) #算x1,x2列差
        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])  #pad是tensor的填充函数，实际上就是为了把X1填充和x2一样大好拼起来
        x=torch.cat([x2,x1],dim=1)  #按照行维度进行连接
        return self.conv(x)  #至此完成上采样，拼接（包括填充过程）即跳跃连接,二次卷积

"""
输出层编码
"""
class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
    def forward(self,x):
        return self.conv(x)

"""
把上述几个部件组合起来就是Unet
"""
class Unet(nn.Module):
    def __init__(self,n_channels,n_classes,biliner=True):
        super(Unet,self).__init__()
        self.n_channels=n_channels  #输入通道数
        self.n_classes=n_classes    #输入类别数
        self.biliner=biliner        #上采样方式

        self.inc=DoubleConv(n_channels,64)  #输入层
        self.down1=Down(64,128)
        self.down2=Down(128,256)
        self.down3=Down(256,512)
        self.down4=Down(512,512)
        #下面这几个数据值得再看一下
        self.up1=Up(1024,256,biliner)
        self.up2=Up(512,128,biliner)
        self.up3=Up(256,64,biliner)
        self.up4=Up(128,64,biliner)
        self.outc=OutConv(64,n_classes)  #输出层

    def forward(self,x):
        x1=self.inc(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x5=self.down4(x4)
        x=self.up1(x5,x4)
        x=self.up2(x,x3)
        x=self.up3(x,x2)
        x=self.up4(x,x1)
        logits=self.outc(x)  #最终输出
        return logits

test_tensor=torch.rand((1,1,572,572),dtype=torch.float32)
Net=Unet(1,2)
print(Net(test_tensor))


