import torch
from torch import nn
input=torch.arange(1,5,dtype=torch.float32).reshape(1,1,2,2)
print(input)

"""
在插值的过程中决定最后输出大小的有两种方式  size方式和scale_factor方式，两者写一个即可否则可能产生冲突

size是最后生成的大小，scale_factor是每一个格子扩展成什么 scale_factor=2就是一个格子变成2*2  scale_factor=(2,3)就是一个格子变成2*3
"""

sample1=nn.Upsample(size=(3,3),mode='nearest')
print(sample1(input))

sample2=nn.Upsample(scale_factor=3,mode='nearest')
print(sample2(input))

sample3=nn.Upsample(scale_factor=(3,2),mode='nearest')
print(sample3(input))
"""
上面的插值方式是nearest，还有bilinear双线性插值方式
"""
sample4=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)  #结合结果看一下吧
print(sample4(input))

sample5=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)   #有点平均插值的意思在里面
print(sample5(input))

sample6=sample5=nn.Upsample(scale_factor=3,mode='bilinear',align_corners=True)
print(sample6(input))