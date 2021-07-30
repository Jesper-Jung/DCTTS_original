import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


''' Text Encoder '''
class TextEnc(nn.Module):
    def __init__(self, num_token, e_dim, dim, causal=False):
        super(TextEnc, self).__init__()

        # character embedder
        self.CharEmbed = nn.Embedding(num_token, e_dim)

        # encoding layers
        enc_layers = []
        enc_layers.append(nn.Conv1d(in_channels=e_dim, out_channels=2*dim, kernel_size=1, stride=1))
        enc_layers.append(nn.ReLU())
        enc_layers.append(nn.Conv1d(in_channels=2*dim, out_channels=2*dim, kernel_size=1, stride=1))
        for i in range(4):
            enc_layers.append(HC(in_dim=2*dim, out_dim=2*dim, kernel=3, dilation=3**i, causal=causal))
        enc_layers.append(HC(in_dim=2*dim, out_dim=2*dim, kernel=3, dilation=1, causal=causal))
        enc_layers.append(HC(in_dim=2*dim, out_dim=2*dim, kernel=1, dilation=1, causal=causal))

        self.enc_layers = nn.Sequential(*enc_layers)
        
    def forward(self, L):
        # L shape : [batch, seq_len]
        # character embedding shape : [batch, channel, seq_len]
        char_embedding = self.CharEmbed(L).permute(0,2,1)
        # encoding procedure
        enc = self.enc_layers(char_embedding)
        # split encoded output into 2
        K, Q = enc.chunk(2, 1)
        return K, Q
  
''' Highway Convolution Module '''
class HC(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, dilation, causal):
        super(HC, self).__init__()
        self.L = nn.Conv1d(in_dim, out_dim*2, kernel_size=kernel, stride=1, dilation = dilation)
        num_pad = dilation * (kernel-1)
        if causal:
            self.paddings = (num_pad, 0)
        else:
            self.paddings = (num_pad//2, num_pad//2)
    def forward(self, x):
        enc = F.pad(x, self.paddings, "constant", 0)
        enc = self.L(enc)
        H1, H2 = enc.chunk(2,1)
        out = torch.sigmoid(H1) * torch.relu(H2) + (1-torch.sigmoid(H1) * x)
        return out

''' Audio Encoder and Decoder '''
# Audio Encoder
class AudioEnc(nn.Module):
    def __init__(self, mel_dim, dim):
        super(AudioEnc, self).__init__()
        self.conv1 = nn.Conv1d(mel_dim, dim, 1, 1)
        self.conv2 = nn.Conv1d(dim, dim, 1, 1)
        self.conv3 = nn.Conv1d(dim, dim, 1, 1)
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(HC(dim, dim, 3, 3**i, causal=True))
        self.layers.append(HC(dim, dim, 3, 3, causal=True))
        self.layers.append(HC(dim, dim, 3, 3, causal=True))

    def forward(self, S):
        enc = self.conv3(F.relu(self.conv2(F.relu(self.conv1(S)))))
        for f in self.layers:
            enc = f(enc)
        Q = enc
        return Q

# Audio Decoder
class AudioDec(nn.Module):
    def __init__(self, mel_dim, dim):
        super(AudioDec, self).__init__()
        self.conv1 = nn.Conv1d(2*dim, dim, 1, 1)
        
        self.layers = nn.ModuleList()
        for i in range(4):
            self.layers.append(HC(dim, dim, 3, 3**i, causal=True))
        self.layers.append(HC(dim, dim, 3, 1, causal=True))
        self.layers.append(HC(dim, dim, 3, 1, causal=True))

        self.conv2 = nn.Conv1d(dim, dim, 1, 1)
        self.conv3 = nn.Conv1d(dim, dim, 1, 1)
        self.conv4 = nn.Conv1d(dim, mel_dim, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, R):
        dec = self.conv1(R)
        for f in self.layers:
            dec = f(dec)
        dec = self.sigmoid(self.conv4(F.relu(self.conv3(F.relu(self.conv2(dec))))))
        return dec


''' Attention Module '''
class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = np.asarray(dim, dtype=np.float32)

    def forward(self, Q, K, V):
        # Q shape : [batch, channel, S-time]
        # K shape : [batch, channel, L-time]
        # V shape : [batch, channel, L-time]
        # -> A shape : [batch, L-time, S-time]
        # -> R shape : [batch, channel, S-time] = V*A
        A = torch.matmul(K.permute(0,2,1), Q) / torch.tensor(self.dim).float().to(Q.device)
        A = torch.softmax(A.float(), dim=1)
        R = torch.matmul(V, A)
        return R, A

if __name__ == "__main__":
    from torchsummaryX import summary
    Module = AudioEnc(80, 128)
    summary(Module, torch.ones(2, 80, 20))

    Module2 = TextEnc(200, 128, 128)
    summary(Module2, torch.arange(150).expand(2, -1).long())

    ### Batch 마다 masking 