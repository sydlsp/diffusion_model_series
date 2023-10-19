import os
import math
from abc import abstractmethod

import torchvision.utils
from PIL import Image
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


"""
对timestep(时间步长)进行编码，采用Attention Is All You Need中所设计的sinusoidal position embeddingm,这里暂时先不看
"""
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    这里先简单理解一下输入输出是什么就好：
    输入是时间步长，输出是我们想要的dim维度的张量，也就是最后编码的结果
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# define TimestepEmbedSequential to support `time_emb` as extra input
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

"""
Unet使用到GroupNorm进行归一化，定义一个简单的norm_layer方法使用
"""
def norm_layer(channels):
    return nn.GroupNorm(32, channels)

"""
Unet核心模块
"""
class  ResidualBlock(TimestepBlock):
    def __init__(self,in_channels,out_channels,time_channels,dropout):
        super().__init__()
        """
        第一次卷积
        """
        self.conv1=nn.Sequential(
            norm_layer(in_channels),  #对输入数据归一化
            nn.SiLU(),  #SiLu激活函数在接近0是有更平滑的曲线，在一些任务中效果可能比ReLu效果更好
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))

        """
        时间步长的嵌入
        """
        self.time_emb=nn.Sequential(nn.SiLU(),
                                    nn.Linear(time_channels,out_channels))

        """
        第二次卷积
        """
        self.conv2=nn.Sequential(norm_layer(out_channels),
                                 nn.SiLU(),
                                 nn.Dropout(p=dropout),
                                 nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))

        if in_channels !=out_channels:
            self.shortcut=nn.Conv2d(in_channels,out_channels,kernel_size=1)  #1*1卷积核额作用是是升降维度，改变了channels的数量大小
        else:
            self.shortcut=nn.Identity()  #在理解了上面的if后这里的else也较好理解了：nn.Identity()是输入是什么直接给输出，说白了就是输入输出通道数一样那1*1卷积核等于没有升降维

    def forward(self,x,t):
        h=self.conv1(x)
        h+=self.time_emb(t)[:,:,None,None]#这里的语法含义是说把编码好的timestep先进行维度扩充再和h相加
        h=self.conv2(h)
        return h+self.shortcut(x)  #问题就是这里加shortcut是干什么的


"""
引入了注意力机制，这里还是先不管了
"""
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

"""
上采样和下采样模块
"""
#上采样
class Upsample(nn.Module):
    def __init__(self,channels,use_conv):
        super().__init__()
        self.use_conv=use_conv
        if use_conv:
            self.conv=nn.Conv2d(channels,channels,kernel_size=3,padding=1)

    def forward(self,x):
        x=F.interpolate(x,scale_factor=2,mode="nearest")  #插值操作
        if self.use_conv:
            x=self.conv(x)
        return x

#下采样
class Downsample(nn.Module):
    def __init__(self,channels,use_conv):
        super().__init__()
        self.use_conv=use_conv
        if use_conv:
            self.op=nn.Conv2d(channels,channels,kernel_size=3,stride=2,padding=1)
        else:
            self.op=nn.AvgPool2d(stride=2)
    def forward(self,x):
        return self.op(x)

"""
组合实现Unet
"""
class UNetModel(nn.Module):
    def __init__(self,in_channels=3,model_channels=128,out_channels=3,num_res_blocks=2,
                 attention_resolutions=(8,16),dropout=0,channel_mult=(1,2,2,2),conv_resample=True,
                 num_heads=4):
        super().__init__()
        self.in_channels=in_channels
        self.model_channels=model_channels
        self.out_channels=out_channels
        self.num_res_blocks=num_res_blocks
        self.attention_resolutions=attention_resolutions
        self.dropout=dropout
        self.channel_mult=channel_mult
        self.conv_resample=conv_resample
        self.num_heads=num_heads


        time_embed_dim=model_channels*4
        self.time_embed=nn.Sequential(nn.Linear(model_channels,time_embed_dim),
                                      nn.SiLU(),
                                      nn.Linear(time_embed_dim,time_embed_dim),)

        #下降模块，我这里倾向于理解下面一大段是实现向下U型左边四个阶段的过程
        self.down_blocks=nn.ModuleList([
            TimestepEmbedSequential(nn.Conv2d(in_channels,model_channels,kernel_size=3,padding=1))])  #这里是什么意思没有怎么看懂
        down_block_chans=[model_channels]
        ch=model_channels
        ds=1
        for level,mult in enumerate(channel_mult): #channel_mult=(1,2,2,2)
            for _ in range(num_res_blocks):  #range(2)
                layers=[ResidualBlock(ch,mult*model_channels,time_embed_dim,dropout)] #ResidualBlock就是前面定义的附加时间信息的两次卷积核心层
                ch=mult*model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch,num_heads=num_heads))  #我这里理解是加上注意力机制
                self.down_blocks.append(TimestepEmbedSequential(*layers))
                down_block_chans.append(ch)
            if level !=len(channel_mult)-1:  #最后一个阶段不需要下采样
                self.down_blocks.append(TimestepEmbedSequential(Downsample(ch,conv_resample)))
                down_block_chans.append(ch)
                ds*=2
        #U中间的那一层
        self.middle_block=TimestepEmbedSequential(
            ResidualBlock(ch,ch,time_embed_dim,dropout),
            AttentionBlock(ch,num_heads=num_heads),
            ResidualBlock(ch,ch,time_embed_dim,dropout)  #这里有的疑问和上面是一样的，就是ResidualBlock已经是两次卷积了为什么还要做两次卷积
        )

        #U右边的四层
        self.up_blocks=nn.ModuleList([])
        for level,mult in list(enumerate(channel_mult))[::-1]:  #[::-1]表示列表逆向取
            for i in range(num_res_blocks+1):
                layers=[ResidualBlock(ch+down_block_chans.pop(),
                        model_channels*mult,
                        time_embed_dim,
                        dropout)]
                ch=model_channels*mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch,num_heads=num_heads))
                if level and i==num_res_blocks:
                    layers.append(Upsample(ch,conv_resample))
                    ds//=2
                self.up_blocks.append(TimestepEmbedSequential(*layers))

        self.out=nn.Sequential(norm_layer(ch),nn.SiLU(),nn.Conv2d(model_channels,out_channels,kernel_size=3,padding=1))
    def forward(self,x,timesteps):
        hs=[]
        emb=self.time_embed(timestep_embedding(timesteps,self.model_channels)) #给时间步编码
        #向下传递过程
        h=x
        for module in self.down_blocks:  #这里一开始把module改成model了
            h=module(h,emb)
            hs.append(h)
        #中间层过程
        h=self.middle_block(h,emb)
        #向上传递过程
        for module in self.up_blocks:
            cat_in=torch.cat([h,hs.pop()],dim=1)
            h=module(cat_in,emb)

        return self.out(h)


"""
上面已经实现了Unet的全部内容，Unet的主要作用是用来预测噪声的分布，注意UNet就是对每一张图预测最原始的那个(0,1)分布的高斯噪声
"""

"""
对于扩散过程，最主要的就是参数在于timesteps和noise schedule(实际上就是数学推到过程中理解的"权重")
"""

def linear_beta_schedule(timesteps):#在DDPM中采用的β范围就是[0.0001,0.02]
    scale=1000/timesteps
    beta_start=scale*0.0001
    beta_end=scale*0.02
    return torch.linspace(beta_start,beta_end,timesteps,dtype=torch.float64)  #这里实际上是生成了一个β序列


"""
定义扩散模型
"""
class GaussianDiffusion:
    def __init__(self,timesteps=1000,beta_schedule='linear'):
        self.timesteps=timesteps
        if beta_schedule=='linear':
            betas=linear_beta_schedule(timesteps)
        # elif beta_schedule=='cosine':
        #     betas=cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        self.betas=betas
        self.alphas=1.-self.betas
        """
        torch.cumprod用法
        x = torch.Tensor([1, 2, 3, 4, 5]) y = torch.cumprod(x, dim = 0)
        y=tensor([ 1., 2., 6., 24., 120.])
        """
        self.alphas_cumprod=torch.cumprod(self.alphas,axis=0) #参考正向过程的推到过程可知需要这个东西
        self.alphas_cumprod_prev=F.pad(self.alphas_cumprod[:-1],(1,0),value=1.)  #也就是说把self.alphas_cumprod除了最后一个元素拿出来在左边扩充了一个1，这儿之前可能打错了


        #这一段都是看变量名，后面的计算公式和变量名是一致的，具体哪里用到了下面再看吧
        self.sqrt_alphas_cumprod=torch.sqrt(self.alphas_cumprod)  #α加根号实际上是加噪的权重，为了满足正态分布的
        self.sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0-self.alphas_cumprod)  #同理，另一个权重
        self.log_one_minus_alphas_cumprod=torch.log(1.0-self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod=torch.sqrt(1.0/self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod=torch.sqrt(1.0/self.alphas_cumprod-1)

        #计算后验概率的均值也就是反向过程的方差和均值
        self.posterior_variance=(self.betas*(1.0-self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod))  #后验方差
        self.posterior_log_variance_clipped=torch.log(self.posterior_variance.clamp(min=1e-20)) #clamp是限制数组在某一个范围内，这里为什么限制并取对数意义还不是很明确
        self.posterior_mean_coef1=(self.betas*torch.sqrt(self.alphas_cumprod_prev)/(1.0-self.alphas_cumprod))  #后验概率x0前面一大块东西，这儿一开始公式打错了
        self.posterior_mean_coef2=(
            (1.0-self.alphas_cumprod_prev)*torch.sqrt(self.alphas)/(1.0-self.alphas_cumprod))         #实际上就是x1前面那一大块东西

    #获取timestep t 的参数，这个还不知道是干什么用的 后面再看，感觉这个函数有点像取前1，2，3，4，5…t个时间步
    def _extract(self,a,t,x_shape):
        batch_size=t.shape[0]
        out=a.to(t.device).gather(0,t).float()
        out=out.reshape(batch_size,*((1,)*(len(x_shape)-1)))
        return out

    #前向扩散过程，也就是求q(x_t|x_0)的过程，照着公式直接求就好了
    def q_sample(self,x_start,t,noise=None):
        if noise is None:
            noise=torch.randn_like(x_start)  #正态分布的噪声

        sqrt_alphas_cumprod_t=self._extract(self.sqrt_alphas_cumprod,t,x_start.shape)  #我觉得对于一张图片来说就是取第t个在sqrt_alphas_cumprod_t的元素，目前是这样理解的
        sqrt_one_minus_alphas_cumprod_t=self._extract(self.sqrt_one_minus_alphas_cumprod,t,x_start.shape)

        return sqrt_alphas_cumprod_t*x_start+sqrt_one_minus_alphas_cumprod_t*noise

    #正向传播的均值和方差，还是看公式，不过这里好像没什么太大的作用，其实和上面前向传播的过程是一样的，具体计算式的实现还是看公式
    def q_mean_variance(self,x_start,t):
        mean=self._extract(self.sqrt_alphas_cumprod,t,x_start.shape)*x_start
        variance=self._extract(1.0-self.alphas_cumprod,t,x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean,variance,log_variance


    #逆过程的均值和方差计算,计算式子还是看公式，不过这里有一部分是在init函数中计算好的，对着看一下就好了
    def q_posterior_mean_variance(self,x_start,x_t,t):
        posterior_mean=(
            self._extract(self.posterior_mean_coef1,t,x_t.shape)*x_start+
            self._extract(self.posterior_mean_coef2,t,x_t.shape)*x_t)

        posterior_variance=self._extract(self.posterior_variance,t,x_t.shape)
        posterior_log_variance_clipped=self._extract(self.posterior_log_variance_clipped,t,x_t.shape)

        return posterior_mean,posterior_variance,posterior_log_variance_clipped



    #根据预测噪声来生成x_0，其实是q_sample的逆过程,下面这个东西算的是什么要好好了解一下
    #在上面的计算过程中我们已经知道了算q(x_t-1|x_t,x_0)用贝叶斯公式，并且得到了均值和方差是什么
    #问题的关键在于最后要求的是x_0但在现在使用的时候还需要用到x_0
    #下面的这个函数就是把x_0用其他方式计算的过程：利用正向传播公式，可以反解出x_0，这个结合正向过程的结论就能知道
    #问题的关键就在于这个noise不知道是多少，所以采用了Unet来预测噪声，逻辑到这里应该说逐渐清晰了
    #下面这个函数实际上就是在代换x_0,求x_0是多少
    def predict_start_from_noise(self,x_t,t,noise):
        return (
                self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    #根据预测的噪音来计算p(x_t-1|x_t)的方差，这个函数是什么意思呢，在上面我们已经计算出了逆过程的均值和方差
    #但上面的函数涉及到x_0没法用，这个函数就是把x_0用上面的函数给他代换了再求逆向过程的均值和方差，这儿我觉得写的有点冗余了但更利于理解了
    def p_mean_variance(self,model,x_t,t,clip_denoised=True): #这儿的model实际上就是Unet，Unet的目的是用来预测噪音
        #利用模型预测噪音
        pred_noise=model(x_t,t)   #这儿就用到了UNet预测的原始噪声
        x_recon=self.predict_start_from_noise(x_t,t,pred_noise)  #这里是算x_0是什么
        if clip_denoised:
            x_recon=torch.clamp(x_recon,min=-1.,max=1.)
            # 下面那个斜杠没什么作用，感觉就是换行的
            # 和model_mean,posterior_variance,posterior_log_variance=self.q_posterior_mean_variance(x_recon,x_t,t是一个意思
        model_mean,posterior_variance,posterior_log_variance=\
                self.q_posterior_mean_variance(x_recon,x_t,t)
        return model_mean,posterior_variance,posterior_log_variance

    #去噪过程：
    @torch.no_grad() #和with torch.no_grad()应该是一样的
    #p_sample是单步去噪过程
    def p_sample(self,model,x_t,t,clip_denoised=True):
        #下面这行代码在计算逆向过程步的均值和方差
        model_mean,_,model_log_variance=self.p_mean_variance(model,x_t,t,clip_denoised=clip_denoised)
        # 这儿要注意了这个noise不是噪声，其作用就是一个(0,1)高斯噪声的底，利用这个底结合均值和方差得到我们想要的高斯分布
        noise=torch.randn_like(x_t)  #填充了均值为0，方差为1的正态分布随机值
        #当t=0的时候没有噪声，应该t!=0的时候nonzero_mask全是1，相当于没有什么变化
        nonzero_mask=((t!=0).float().view(-1,*([1]*(len(x_t.shape)-1))))  #括号前加*表示解包
        #计算x_t-1，这里可以用笔展开来计算一下实际上就是把(0,1)的正态分布变成我们所需要的正态分布
        pred_img=model_mean+nonzero_mask*(0.5*model_log_variance).exp()*noise
        return pred_img

    @torch.no_grad()
    #整体去噪过程，也就是生成过程
    def p_sample_loop(self,model,shape):
        batch_size=shape[0]
        device=next(model.parameters()).device
        #从最纯的(0,1)正态分布噪声开始
        img=torch.randn(shape,device=device)
        imgs=[]
        for i in tqdm(reversed(range(0,self.timesteps)),desc='sampling loop time step',total=self.timesteps):
            img = self.p_sample(model, img, torch.full((batch_size,), i, device=device, dtype=torch.long))
            imgs.append(img.cpu().numpy())
        return imgs

    @torch.no_grad()
    def sample(self,model,image_size,batch_size=8,channels=3):
        return self.p_sample_loop(model,shape=(batch_size,channels,image_size,image_size))

    #计算训练损失,从损失函数看Unet预测的是什么
    def train_losses(self,model,x_start,t):
        #下面这个噪声就是说正向加的是什么，UNet预测的就是什么
        noise=torch.randn_like(x_start)
        x_noisy=self.q_sample(x_start,t,noise=noise)  #q_sample 是在第t步加噪完的结果
        predicted_noise=model(x_noisy,t)
        loss=F.mse_loss(noise,predicted_noise)  #就是这里存在一个小问题
        #2023.10.19  这个小问题好像有了一个答案：在正向传播过程中我加了一个噪声noise满足(0,1)分布的原始高斯噪声
        #下面我要干的事情是我在反推x_0的时候需要知道这个原始噪声是多少
        #下面的过程就是我用UNet对每一张图片都要预测出这个原始噪声，进而反推UNet
        return loss


"""
扩散过程展示
"""
# image=Image.open("000000039769.jpg")
# image_size=128
# #下面这个transfrom简单理解就是对图片进行修改，尺寸等等
# transform=transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.CenterCrop(image_size),
#     transforms.PILToTensor(),
#     transforms.ConvertImageDtype(torch.float),
#     transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
# ])
# x_start=transform(image).unsqueeze(0)  #unsqueeze(0)表示在第0维进行升维，这儿不是很明确为什么要升维
#
# gaussian_diffusion=GaussianDiffusion(timesteps=500)  #看一下init需要什么参数
# plt.figure(figsize=(16,8))
# for idx,t in enumerate([0,50,100,200,499]):
#     x_noisy=gaussian_diffusion.q_sample(x_start,t=torch.tensor([t]))  #对图片进行加噪，加噪过程是很快的，因为可以一步到位
#     noisy_image=(x_noisy.squeeze().permute(1,2,0)+1)*127.5  #squeeze函数就是unsqueeze的逆过程,permute函数是将tensor的维度换位
#     noisy_image=noisy_image.numpy().astype(np.uint8)
#     plt.subplot(1, 5, 1 + idx)
#     plt.imshow(noisy_image)
#     plt.axis("off")
#     plt.title(f"t={t}")
# plt.show()


"""
针对minst进行生成
"""
batch_size=64
timesteps=500
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])


"""
加载minst数据集
"""
dataset=datasets.MNIST('./data',train=True,download=True,transform=transform)
train_loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

device="cuda:0" if torch.cuda.is_available() else "cpu"
model=UNetModel(
    in_channels=1,model_channels=96,out_channels=1,channel_mult=(1,2,2),
    attention_resolutions=[]
)
model.load_state_dict(torch.load('model_2epoches_500steps_new.params'))
model.to(device=device)


gaussion_diffusion=GaussianDiffusion(timesteps=timesteps)
optimizer=torch.optim.Adam(model.parameters(),lr=5e-4)


"""
训练过程，这里是训练Unet的参数
"""
# epoches=2
# for epoch in range(epoches):
#     print("epoch:",epoch)
#     for step,(images,labels) in enumerate(train_loader):
#         optimizer.zero_grad()
#
#         batch_size=images.shape[0]
#         images=images.to(device)
#         t = torch.randint(0, timesteps, (batch_size,), device=device).long()  #这里实际上生成了一个t的随机序列
#         loss=gaussion_diffusion.train_losses(model,images,t)
#
#         if step %200==0:
#             print("Loss:",loss.item())
#         loss.backward()
#         optimizer.step()
#
# torch.save(model.state_dict(),'model_2epoches_500steps_new.params')

"""
生成图片
"""
generated_images=gaussion_diffusion.sample(model,28,batch_size=64,channels=1)



"""
最终结果展示
"""
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(8, 8)

imgs = generated_images[-1].reshape(8, 8, 28, 28)
for n_row in range(8):
    for n_col in range(8):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        f_ax.imshow((imgs[n_row, n_col]+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")
plt.show()
"""
生成过程展示
"""
fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(16, 16)

for n_row in range(16):
    for n_col in range(16):
        f_ax = fig.add_subplot(gs[n_row, n_col])
        t_idx = (timesteps // 16) * n_col if n_col < 15 else -1
        img = generated_images[t_idx][n_row].reshape(28, 28)
        f_ax.imshow((img+1.0) * 255 / 2, cmap="gray")
        f_ax.axis("off")
plt.show()







































































