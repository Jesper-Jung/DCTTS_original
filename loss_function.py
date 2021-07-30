import torch
from torch.nn.modules.loss import _Loss

import numpy as np

class Binary_Divergence_Loss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, predict_mel, target_mel):
        return binary_divergence_loss(predict_mel, target_mel)

class Guided_Attention_Loss(_Loss):
    def __init__(self):
        super().__init__()

    def forward(self, attn_map, scr_len, mel_len):
        return guided_attention_loss(attn_map, scr_len, mel_len)




def binary_divergence_loss(predict_mel, target_mel):
    # ! Parameters
    # ? Input
    # predict_mel: predicted mel spectrogram  || (batch, n_mel, frame)
    # target_mel: target mel spectrogram      || (batch, n_mel, frame)

    eps = torch.finfo(torch.float32).eps
    
    target_mel = torch.clamp(target_mel, 0, 1)
    predict_mel = torch.clamp(predict_mel, 0, 1)

    first_term = - target_mel * torch.log(torch.div(predict_mel, target_mel + eps) + eps)
    second_term = - (1 - target_mel) * torch.log(torch.div((1 - predict_mel), (1 - target_mel + eps)) + eps)

    return torch.mean(first_term + second_term)


def guided_attention_loss(attn_map, scr_len, mel_len):
    # ! Parameters
    # ? Input
    # attn_map: attention matrix        || (batch, max_scr_len, max_mel_len)
    # scr_len: text length              || (batch, ), int
    # mel_len: mel frame length         || (batch, ), int

    batch_size, max_scr_len, max_mel_len = attn_map.shape

    # calculate gaussian kernel
    gaussian_kernel = [_pad2d(_gaussian_kernel(scr_len[i].item(), mel_len[i].item()), max_scr_len, max_mel_len)\
         for i in range(batch_size)]
    gaussian_kernel = torch.tensor(gaussian_kernel) # (batch, max_scr_len, max_mel_len)

    return torch.mean(attn_map * gaussian_kernel)


def _gaussian_kernel(scr_len, mel_len):
    """ return gaussian kernel for GA_loss""" 
    g = 0.2 # temperature coefficient

    # Calculate
    W = np.arange(scr_len)[:, np.newaxis] / scr_len - np.arange(mel_len)[np.newaxis, :] / mel_len
    W = 1 - np.exp(-W ** 2 / 2 / (g ** 2))

    return W

def _pad2d(matrix, max_scr_len, max_mel_len):
    W = np.zeros((max_scr_len, max_mel_len))
    W[ :matrix.shape[0], :matrix.shape[1]] = matrix
    
    return W





if __name__ == "__main__":
    # test code
    a = torch.randn(2, 5, 6)
    scr_len = torch.tensor([1, 4])
    mel_len = torch.tensor([3, 4])
    b = torch.randn(2, 80, 89)
    print(guided_attention_loss(a, scr_len, mel_len))


