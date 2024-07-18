import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def temperature_scaled_softmax(inputs, temperature=1.0):
    scaled_inputs = inputs / temperature
    softmax_outputs = F.softmax(scaled_inputs, dim=-1)
    return softmax_outputs



def gloss_level_alignment_loss(text, visual,label_lgt):

    text = text.to(dtype=torch.float32)  
    visual = visual.to(dtype=torch.float32)  
    text = torch.nn.functional.normalize(text, p=2, dim=2)
    visual = torch.nn.functional.normalize(visual, p=2, dim=2)
    a = torch.matmul(text, visual.permute(0, 2, 1))
    b = torch.matmul(visual, text.permute(0, 2, 1))
    temperature = 1.0
    T2V_softmax = temperature_scaled_softmax(a, temperature)
    V2T_softmax = temperature_scaled_softmax(b, temperature)
    V2T_mod = V2T_softmax 
    T2V_mod = T2V_softmax
    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    for label_lgts, v2tmods, t2vmods in zip(label_lgt, V2T_mod, T2V_mod):

        G = torch.arange(label_lgts)
        G = G.to('cuda')
        loss_V2T = loss_fn(v2tmods[0:label_lgts,0:label_lgts], G)        
        loss_T2V = loss_fn(t2vmods[0:label_lgts,0:label_lgts], G)
        loss += 0.5 * (loss_V2T + loss_T2V)
    return loss

