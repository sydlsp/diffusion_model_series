import torch
from torch import nn

"""
gather函数实际上就是从tensor中取数的过程
来自知乎的解释，先简单这样理解一下吧
torch.gather的理解
index=[ [x1,x2,x2],
[y1,y2,y2],
[z1,z2,z3] ]

如果dim=0
填入方式
[ [(x1,0),(x2,1),(x3,2)]
[(y1,0),(y2,1),(y3,2)]
[(z1,0),(z2,1),(z3,2)] ]

如果dim=1
[ [(0,x1),(0,x2),(0,x3)]
[(1,y1),(1,y2),(1,y3)]
[(2,z1),(2,z2),(2,z3)] ]
"""

tensor_0=torch.arange(3,12).view(3,3)
print(tensor_0)

index=torch.tensor([[2,1,1]])

#gather函数实际上是按照dim取得tensor_0在index中的元素并输出，dim=0是按照列来取，dim=1是按照行来取
tensor_1=tensor_0.gather(0,index)
print(tensor_1)

tensor_2=tensor_0.gather(1,index)
print(tensor_2)

"""
另一个方向
"""
index_1=index.reshape(3,-1)
tensor_3=tensor_0.gather(0,index_1)
print(tensor_1)

tensor_4=tensor_0.gather(1,index_1)
print(tensor_4)
