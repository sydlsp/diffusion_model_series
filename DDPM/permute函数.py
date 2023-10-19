import torch
x = torch.randn(2, 3, 5)
print(x.size())
print(x.permute(2, 0, 1).size())





