import torch
import torch.nn as nn
import torch.nn.functional as F

import yaml
import os
from Network.module import *

class DCTTS(nn.Module):
    def __init__(self, config):
        super().__init__()
        max_txt_token = config["Model"]["max_txt_token"]
        e_dim = config["Model"]["embed_txt_dim"]
        dim = config["Model"]["attention_dim"]
        mel_dim = config["Model"]["mel_dim"]

        self.TextEnc = TextEnc(max_txt_token, e_dim, dim)
        self.AudioEnc = AudioEnc(mel_dim, dim)
        self.Attention = Attention(dim)
        self.AudioDec = AudioDec(mel_dim, dim)

    def forward(self, S, L):
        K, V = self.TextEnc(L)
        Q = self.AudioEnc(S)
        R, A = self.Attention(Q, K, V)
        R = torch.cat([R, Q], axis=1)
        Y = self.AudioDec(R)

        if not self.training:
            self.attention_map = A

        return Y
        
if __name__ == "__main__":
    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        model = DCTTS(config)

        from torchsummaryX import summary
        Spec = torch.randn(2, 80, 128).float()
        Txt = torch.arange(20).contiguous().reshape(2, 10)
        summary(model, Spec, L=Txt)

