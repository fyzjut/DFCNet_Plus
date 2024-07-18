import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from transformers import MBartTokenizerFast, MBartModel
from loss_clip import gloss_level_alignment_loss
from thop import profile




def seq_train(loader, model, optimizer, device, epoch_idx, recoder,tokenizer,mbartencode):
    model.train()
  
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    scaler = GradScaler()

    for batch_idx, data in enumerate(tqdm(loader)):
        a = data[5]
        text = []
        for i in range(len(a)):
            text.append(a[i]['label'])
        #print(text)
        gloss_index_batch = []
 
        encoded_inputs = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True)
        offset_mappings = encoded_inputs["offset_mapping"]

        gloss_index_batches = []
        for offset_mapping in offset_mappings:
            gloss_index = []
            k, u = 1, 0
            ori = 0
            for i, (start, end) in enumerate(offset_mapping):
                if start != ori:
                    u = i
                    if u >= k:
                        gloss_index.append((k, u))
                        if start != offset_mapping[i + 1][0] or start == 0:
                            k = i + 1
                        else:
                            k = i + 2
                ori = end
            gloss_index_batches.append(torch.tensor(gloss_index))
        

        gloss_index_batches = [device.data_to_device(gloss_index) for gloss_index in gloss_index_batches]
 
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs.get("attention_mask", None)  
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda') 
        with torch.no_grad():
            outputs =  mbartencode.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state

        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        optimizer.zero_grad()
        with autocast():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt , gloss_index_batch=gloss_index_batches, last_hidden_states=last_hidden_states)
       
            loss = model.criterion_calculation(ret_dict, label, label_lgt)
            
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print('loss is nan')
            print(str(data[1])+'  frames')
            print(str(data[3])+'  glosses')
            continue
		   
        scaler.scale(loss).backward()
        scaler.step(optimizer.optimizer)
        scaler.update()
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.10f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
        del ret_dict
        del loss
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return 

def cal_torch_model_params(model):
    '''
    :param model:
    :return:
    '''
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total_params': total_params/10000, 'total_trainable_params': total_trainable_params/10000}

def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,tokenizer,mbartencode,
             evaluate_tool="python"):

    model.eval()

    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        a = data[5]

        text = []
        for i in range(len(a)):
            text.append(a[i]['label'])
        #print(text)
        gloss_index_batch = []
        
        encoded_inputs = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True)
        offset_mappings = encoded_inputs["offset_mapping"]

        gloss_index_batches = []
        for offset_mapping in offset_mappings:
            gloss_index = []
            k, u = 1, 0
            ori = 0
            for i, (start, end) in enumerate(offset_mapping):
                if start != ori:
                    u = i
                    if u >= k:
                        gloss_index.append((k, u))
                        if start != offset_mapping[i + 1][0] or start == 0:
                            k = i + 1
                        else:
                            k = i + 2
                ori = end
            gloss_index_batches.append(torch.tensor(gloss_index))
        

        gloss_index_batches = [device.data_to_device(gloss_index) for gloss_index in gloss_index_batches]
    
        encoded_input = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded_input["input_ids"]
        
        input_ids = device.data_to_device(input_ids)
        last_hidden_states = []
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input.get("attention_mask", None)  
        input_ids = input_ids.to('cuda')
        attention_mask = attention_mask.to('cuda')
        with torch.no_grad():
            outputs =  mbartencode.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state



        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])

        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt , gloss_index_batch=gloss_index_batches, last_hidden_states=last_hidden_states)
            
        total_info += [file_name.split("|")[0] for file_name in data[4]]
        total_sent += ret_dict['recognized_sents']
        total_conv_sent += ret_dict['conv_sents']
    try:
        python_eval = True if evaluate_tool == "python" else False
        write2file(work_dir + "output-hypothesis-{}.ctm".format(mode), total_info, total_sent)
        write2file(work_dir + "output-hypothesis-{}-conv.ctm".format(mode), total_info,
                   total_conv_sent)
        conv_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}-conv.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
        )
        lstm_ret = evaluate(
            prefix=work_dir, mode=mode, output_file="output-hypothesis-{}.ctm".format(mode),
            evaluate_dir=cfg.dataset_info['evaluation_dir'],
            evaluate_prefix=cfg.dataset_info['evaluation_prefix'],
            output_dir="epoch_{}_result/".format(epoch),
            python_evaluate=python_eval,
            triplet=True,
        )
    except:
        print("Unexpected error:", sys.exc_info()[0])
        lstm_ret = 100.0
    finally:
        pass
    del conv_ret
    del total_sent
    del total_info
    del total_conv_sent
    del vid
    del vid_lgt
    del label
    del label_lgt
    # del space_positions
    del offset_mapping
    del gloss_index
    del gloss_index_batch
    del last_hidden_states
    del outputs
    del attention_mask
    del encoded_input
    recoder.print_log(f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt")
    return lstm_ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[4][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))


