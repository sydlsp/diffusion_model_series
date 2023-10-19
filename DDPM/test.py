import torch
import torch.nn.functional as F
x = torch.Tensor([1, 2, 3, 4, 5])
y = torch.cumprod(x, dim = 0)
print(y)
z=F.pad(y[0:-1],(2,3),value=1.)
print(z)
size=2
print(1.0/size-1)