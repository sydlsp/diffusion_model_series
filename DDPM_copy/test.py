import torch

timesteps=10
batch_size=16
device="cuda:0"
t = torch.randint(0, timesteps, (batch_size,), device=device).long()
print(t)
print("ok")
