import torch
input=torch.arange(0,6)
print(input)
print(input.shape)
print('_'*100)
print(input.unsqueeze(0))
print(input.unsqueeze(0).shape)
print('_'*100)
print(input.unsqueeze(1))
print(input.unsqueeze(1).shape)






