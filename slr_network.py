import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet
from DTW import DTW
from loss_clip import gloss_level_alignment_loss


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs



class WeightedResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(WeightedResidualBlock, self).__init__()
        self.fc1 = NormLinear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout_rate)  # 添加Dropout层
        self.fc2 = NormLinear(out_features, out_features)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        alpha = F.sigmoid(self.alpha)
        identity = x
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout(out)  
        out = self.fc2(out)
        out = F.relu(out)
        out = (1.0 - self.alpha) * identity + self.alpha * out
        return out





class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, 'resnet34')()
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

        self.V2TAdapter = WeightedResidualBlock(hidden_size, hidden_size)
        self.T2VAdapter = WeightedResidualBlock(hidden_size, hidden_size)




    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x
    
    def add_first_row_after_each_row(self, tensor):
        first_row = tensor[0:1, :]  
        new_rows = [first_row]  
        for i in range(1, tensor.size(0)):
            row = tensor[i:i+1, :]  
            new_rows.extend([row, first_row])  
        result_tensor = torch.cat(new_rows, dim=0)
        return result_tensor

    def extract_feat(self, x, label, label_lgt, lgt):
       # print(label)
        label_lgt = label_lgt.tolist()
        extracted_list = []
        x = x.permute(1,0,2)
        label_lgt.insert(0,0)
        if len(label_lgt) == 3:
            label_lgt[2] = label_lgt[2] + label_lgt[1]
        for i in range(x.size(0)):  
            extracted_rows = []            
            blank_row= x[i, :, 0]
            extracted_rows.append(blank_row.unsqueeze(0))
            for j in range(label_lgt[i],label_lgt[i+1]):  
                idx = label[j]  
                extracted_row = x[i, :,idx]  # 提取对应的行
                extracted_rows.append(extracted_row.unsqueeze(0))
              
            extracted_tensor = torch.cat(extracted_rows, dim=0)
            extracted_tensor.permute(1,0)
            extracted_tensor = F.softmax(extracted_tensor, dim=0)
            extracted_tensor = self.add_first_row_after_each_row(extracted_tensor)  
            a = lgt[i].item()
            extracted_tensor = extracted_tensor[:,0:int(a)]
            extracted_list.append( extracted_tensor)
        return extracted_list

    def mbart_pooling(self,text_feat,gloss_index_batch):
        pooled_tensor = []
        max_len = 0
        for hidden_state, gloss_index in zip(text_feat, gloss_index_batch):
            pooled_features = []
            for start, end in gloss_index:
                pooled = hidden_state[(start-1):(end),:].mean(dim=0, keepdim=True)
                pooled_features.append(pooled)                
            pooled_features = torch.cat(pooled_features, dim=0)
            max_len = max(max_len, pooled_features.size(0))
            pooled_tensor.append(pooled_features)
        final_features = []
        for feat in pooled_tensor:
            if feat.size(0) < max_len:
                pad_size = max_len - feat.size(0)
                padded_feat = torch.nn.functional.pad(feat, (0, 0,  0, pad_size), "constant", 0)
            else:
                padded_feat = feat
            final_features.append(padded_feat)
        batch_tensor = torch.stack(final_features, dim=0)
        return batch_tensor

    def forward(self, x, len_x, label=None, label_lgt=None,gloss_index_batch=None,last_hidden_states = None):
      
        pooled_text_feat = self.T2VAdapter(last_hidden_states)
        pooled_text_feat = self.mbart_pooling(pooled_text_feat, gloss_index_batch)
        
        last_hidden_states = None
        text_feat = None
        gloss_index_batch = None
        

        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            #inputs = x.reshape(batch * temp, channel, height, width)
            #framewise = self.masked_bn(inputs, len_x)
            #framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
            torch.cuda.empty_cache()
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
            x = None
        else:
            framewise = x
        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
     
        outputs[:, :, 0] /= 3.

        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        
        

        visual_feat = x.permute(1,0,2)
        visual_feat = self.V2TAdapter(visual_feat)

        extracted_outputs = self.extract_feat(outputs, label, label_lgt,lgt)
        averaged_feats_tensor = []
        max_len = max(label_lgt)
        for i in range(len(extracted_outputs)):
            path,DTW_index = DTW(extracted_outputs[i])
            visual_feat_bat = visual_feat[i]
            averaged_feats = []  
            start_idx = 0  
            for idx in DTW_index:
                start, end = idx
                group_avg = visual_feat_bat[start: end+1, : ].mean(dim=0,keepdim=True)  # 计算每行在指定列范围内的平均值
                averaged_feats.append(group_avg)  

            averaged_feats_tensor.append(torch.cat(averaged_feats, dim=0))
        extracted_outputs = None
        padded_feat_list = []

        for average_feat in averaged_feats_tensor:
  
            if len(average_feat)<max_len:
                pad_size = max_len - len(average_feat)
                #padded_feat = feat
                padded_average_feat = torch.nn.functional.pad(average_feat, (0, 0,  0, pad_size), "constant", 0)
                padded_feat_list.append(padded_average_feat)
            else:
                padded_average_feat = average_feat
                padded_feat_list.append(padded_average_feat)
       
        averaged_feats_tensor = torch.stack(padded_feat_list, dim=0)
        



       
        return {
            #"framewise_features": framewise,
            #"visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
            "pooled_text_feat":pooled_text_feat,
            "averaged_feats_tensor":averaged_feats_tensor,
            "loss_LiftPool_u": conv1d_outputs['loss_LiftPool_u'],
            "loss_LiftPool_p": conv1d_outputs['loss_LiftPool_p'],
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
         
                CTCLOSS = self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int())                                        
                loss += weight * CTCLOSS.mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
            elif k == 'Clip':
                #loss += 1
                loss += weight * gloss_level_alignment_loss(ret_dict["pooled_text_feat"],ret_dict["averaged_feats_tensor"],label_lgt.int()).cpu()
            elif k == 'Cu':
                loss += weight * ret_dict["loss_LiftPool_u"]
            elif k == 'Cp':
                loss += weight * ret_dict["loss_LiftPool_p"] 
        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
