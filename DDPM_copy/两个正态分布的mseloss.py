import torch
from torch import nn

test=torch.arange(0,10.0).reshape(2,5)
for i in range(10):
    rand_A=torch.randn_like(test)
    rand_B=torch.randn_like(test)
    print(rand_A==rand_B)
    loss=nn.MSELoss()
    print(loss(rand_A,rand_B))  #组会时问一下吧，有点想不明白了
