import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd:int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, 4*n_embd)
        

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320,4)
    
    def forward(self, latent:torch.Tensor, context:torch.Tensor, time: torch.Tensor):
        #latent: bsize, 4, height/8, width/8
        #context: bsize, seqlen, dim
        #time: (1,320)

        # (1,320) -> (1,1280)
        time = self.time_embedding(time)

        #(bsize, 4, height / 8, width/8) -> bsize, 320, height/8, width/8
        output = self.unet(latent, context, time)

        #bsize, 320, height/8, width/8 -> bsize, 4, height/8, width/8
        output = self.final(output)

        return output

