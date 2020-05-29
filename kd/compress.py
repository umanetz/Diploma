import os
import os.path
import numpy as np
import pandas as pd
import argparse

import torch
from torch import nn
from model import StackBiLSTM, StackLSTM
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_path",  default=None)
    parser.add_argument("--show_weight", action='store_true')
    parser.add_argument("--cut_model", action='store_true')
    args = parser.parse_args()
    return args

ARGS = get_args()


class DistilModelConfigBiLSTM(object):
    """Конфигурация для модели stack_biLSTM"""
    def __init__(self):
        self.init_scale = 0.04
        self.hidden_size1 = ARGS.hidden_state
        self.hidden_size2 = ARGS.hidden_state
        self.keep_prob = 0.60
        self.output_size = 80
        self.bi = True
        self.embedding_dim = ARGS.hidden_state
        self.vocab_size = 32530
        
        
class DistilModelConfigLSTM(object):
    """Конфигурация для модели stack_LSTM"""
    def __init__(self):
        self.init_scale = 0.04
        self.hidden_size1 = ARGS.hidden_state
        self.hidden_size2 = ARGS.hidden_state
        self.hidden_size3 = ARGS.hidden_state
        self.hidden_size4 = ARGS.hidden_state
        self.keep_prob = 0.60
        self.output_size = 80
        self.bi = False
        self.embedding_dim = ARGS.hidden_state
        self.vocab_size = 32530
        
        
def extraxt_matrix_LSTM(model):
    ''' извлекаем матрицы весов LSTM'''
    matrix = {}

    hs = model.hidden_dim1
    var = torch.cat([model.lstm1.weight_ih_l0.T, model.lstm1.weight_hh_l0.T]).cpu().data.numpy()
    couple_var = torch.cat([model.lstm3.weight_ih_l0.T[:hs], model.lstm4.weight_ih_l0.T[:hs]], -1).cpu().data.numpy()
    matrix['lstm1'] = (var, couple_var)
    
    hs = model.hidden_dim2
    var = torch.cat([model.lstm2.weight_ih_l0.T, model.lstm2.weight_hh_l0.T]).cpu().data.numpy()
    couple_var = torch.cat([model.lstm3.weight_ih_l0.T[-hs:], model.lstm4.weight_ih_l0.T[-hs:]], -1).cpu().data.numpy()
    matrix['lstm2'] = (var, couple_var)

    hs = model.hidden_dim3
    var = torch.cat([model.lstm3.weight_ih_l0.T, model.lstm3.weight_hh_l0.T]).cpu().data.numpy()
    couple_var = model.softmax.weight.T[:hs].cpu().data.numpy()
    matrix['lstm3'] = (var, couple_var)

    hs = model.hidden_dim4
    var = torch.cat([model.lstm4.weight_ih_l0.T, model.lstm4.weight_hh_l0.T]).cpu().data.numpy()
    couple_var = model.softmax.weight.T[-hs:].cpu().data.numpy()
    matrix['lstm4'] = (var, couple_var)
    return matrix


def extraxt_matrix_biLSTM(model):
    ''' извлекаем матрицы весов biLSTM'''
    matrix = {}
    var = torch.cat([model.lstm1.weight_ih_l0.T, model.lstm1.weight_hh_l0.T]).cpu().data.numpy()
    var_revers = torch.cat([model.lstm1.weight_ih_l0_reverse.T, model.lstm1.weight_hh_l0_reverse.T]).cpu().data.numpy()
    couple_var = torch.cat([model.lstm2.weight_ih_l0.T, model.lstm2.weight_ih_l0_reverse.T], 1).cpu().data.numpy()

    var = np.hstack([var, var_revers])

    matrix['lstm1'] = (var, couple_var)
    
    var = torch.cat([model.lstm2.weight_ih_l0.T, model.lstm2.weight_hh_l0.T]).cpu().data.numpy()
    var_revers = torch.cat([model.lstm2.weight_ih_l0_reverse.T, model.lstm2.weight_hh_l0_reverse.T]).cpu().data.numpy()
    couple_var = model.softmax.weight.T.cpu().data.numpy()

    var = np.hstack([var, var_revers])

    matrix['lstm2'] = (var, couple_var)
    return matrix


def extract_iss(t, coupled_t, title, model_type, split_num=2):
    ''' извлекаем индексы компонент (строки, столбцы), которые будут удалены из матриц весов LSTM'''
    print(title)
    col_zero_idx = np.sum(np.abs(t), axis=0) == 0      
    row_zero_idx = np.sum(np.abs(t), axis=1) == 0 
    if coupled_t is not None:
        coupled_row_zero_idx = np.sum(np.abs(coupled_t), axis=1) == 0    

    match_idx = None
    if coupled_t is not None:
        if model_type == 'stack_biLSTM':
            subsize = int(t.shape[0] // split_num)
            match_map = np.zeros(subsize, dtype=np.int)
            match_map = match_map + row_zero_idx[-subsize:]

            for blk in range(0,2):
                match_map = match_map + coupled_row_zero_idx[blk*subsize:(blk+1)*subsize]
            for blk in range(0,8):
                match_map = match_map + col_zero_idx[blk*subsize : blk*subsize+subsize]
            match_idx = np.where(match_map == 11)[0]
        elif model_type == 'stack_LSTM':
            subsize = int(t.shape[0] - split_num)
            match_map = np.zeros(subsize, dtype=np.int)
            match_map = match_map + row_zero_idx[-subsize:]
            match_map = match_map + coupled_row_zero_idx

            for blk in range(0,4):
                match_map = match_map + col_zero_idx[blk*subsize : blk*subsize+subsize]
            match_idx = np.where(match_map == 6)[0]
        print('Кол-во нулевых компонент', len(match_idx))
    return match_idx


def restructur_biLSTM(model, matrix, config, coupled_iss):
    '''создаем новую сокращенную архитектуру для модели типа stack_biLSTM'''
    coupled_iss1, coupled_iss2 = coupled_iss
    new_weight = {'emb.weight': model.emb.weight}
    
    var, _ = matrix['lstm1']
    row_index1 = np.array([x for x in range(config.hidden_size1) if x not in coupled_iss1])
    input_W, hidden_W = np.split(var, 2, axis=0)
        
    col_index1 = []
    for i in range(8):
        col_index1.extend(row_index1 + i*config.hidden_size1)

    forward_W_I, reverse_W_I = np.split(input_W[:, col_index1], 2, axis=1)
    forward_W_H, reverse_W_H = np.split(hidden_W[row_index1][:, col_index1], 2, axis=1)

    new_weight['lstm1.weight_ih_l0'] = torch.tensor(forward_W_I.T).contiguous()
    new_weight['lstm1.weight_hh_l0'] = torch.tensor(forward_W_H.T).contiguous()
    new_weight['lstm1.weight_ih_l0_reverse'] = torch.tensor(reverse_W_I.T).contiguous()
    new_weight['lstm1.weight_hh_l0_reverse'] = torch.tensor(reverse_W_H.T).contiguous()

    bias_index = []
    for i in range(4):
        bias_index.extend(row_index1 + i*config.hidden_size1)

    for n, m in model.lstm1.named_parameters():
        if 'bias' in n:
            new_weight['lstm1.'+n] = m[bias_index].data.contiguous()

    var, cls = matrix['lstm2']
    row_index2 = np.array([x for x in range(config.hidden_size2) if x not in coupled_iss2])
    input_W, hidden_W = var[:config.hidden_size2*2], var[config.hidden_size2*2:]

    row_index12 = np.append(row_index1, row_index1 + config.hidden_size2)
    col_index2 = []
    for i in range(8):
        col_index2.extend(row_index2 + i*config.hidden_size2)

    forward_W_I, reverse_W_I = np.split(input_W[row_index12, :][:, col_index2], 2, axis=1)
    forward_W_H, reverse_W_H = np.split(hidden_W[row_index2][:, col_index2], 2, axis=1)

    new_weight['lstm2.weight_ih_l0'] = torch.tensor(forward_W_I.T).contiguous()
    new_weight['lstm2.weight_hh_l0'] = torch.tensor(forward_W_H.T).contiguous()
    new_weight['lstm2.weight_ih_l0_reverse'] = torch.tensor(reverse_W_I.T).contiguous()
    new_weight['lstm2.weight_hh_l0_reverse'] = torch.tensor(reverse_W_H.T).contiguous()

    bias_index = []
    for i in range(4):
        bias_index.extend(row_index2 + i*config.hidden_size2)

    for n, m in model.lstm2.named_parameters():
        if 'bias' in n:
            new_weight['lstm2.'+n] = m[bias_index].data.contiguous()

    row_index23 = np.append(row_index2, row_index2 + config.hidden_size2)
    cls = cls[row_index23]
    new_weight['softmax.weight'] = torch.tensor(cls.T).contiguous()
    new_weight['softmax.bias'] = model.softmax.bias.cpu().data.contiguous()
    
    
    config.hidden_size1 = len(row_index1)
    config.hidden_size2 =  len(row_index2)

    print('\nNEW DIMS')
    print('hidden_size1', len(row_index1))
    print('hidden_size2', len(row_index2))
    return new_weight, config


def restructur_LSTM(model, config, coupled_iss):
    '''создаем новую сокращенную архитектуру для модели типа stack_LSTM'''
    coupled_iss1, coupled_iss2, coupled_iss3, coupled_iss4 = coupled_iss
    new_weight = {'emb.weight': model.emb.weight}
    row_index1 = np.array([x for x in range(config.hidden_size1) if x not in coupled_iss1])
    row_index2 = np.array([x for x in range(config.hidden_size1) if x not in coupled_iss2])
    
    input_W1 = model.lstm1.weight_ih_l0.data.T
    hidden_W1 = model.lstm1.weight_hh_l0.data.T
    input_W2 = model.lstm2.weight_ih_l0.data.T
    hidden_W2 = model.lstm2.weight_hh_l0.data.T
    
    col_index1 = []
    for i in range(4):
        col_index1.extend(row_index1 + i*config.hidden_size1)

    col_index2 = []
    for i in range(4):
        col_index2.extend(row_index2 + i*config.hidden_size2)

    forward_W_I1 = input_W1[:, col_index1]
    forward_W_H1 = hidden_W1[row_index1][:, col_index1]

    forward_W_I2 = input_W2[:, col_index2]
    forward_W_H2 = hidden_W2[row_index2][:, col_index2]

    new_weight['lstm1.weight_ih_l0'] = forward_W_I1.T.contiguous()
    new_weight['lstm1.weight_hh_l0'] = forward_W_H1.T.contiguous()
    new_weight['lstm2.weight_ih_l0'] = forward_W_I2.T.contiguous()
    new_weight['lstm2.weight_hh_l0'] = forward_W_H2.T.contiguous()

    for n, m in model.lstm1.named_parameters():
        if 'bias' in n:
            new_weight['lstm1.'+n] = m.data[col_index1].contiguous()
    for n, m in model.lstm2.named_parameters():
        if 'bias' in n:
            new_weight['lstm2.'+n] = m.data[col_index2].contiguous()

    row_index_input34 = np.append(row_index1, row_index2 + model.hidden_dim1)
    
    row_index3 = np.array([x for x in range(config.hidden_size1) if x not in coupled_iss3])
    row_index4 = np.array([x for x in range(config.hidden_size1) if x not in coupled_iss4])

    input_W3 = model.lstm3.weight_ih_l0.data.T
    hidden_W3 = model.lstm3.weight_hh_l0.data.T
    input_W4 = model.lstm4.weight_ih_l0.data.T
    hidden_W4 = model.lstm4.weight_hh_l0.data.T

    col_index3 = []
    for i in range(4):
        col_index3.extend(row_index3 + i*config.hidden_size3)

    col_index4 = []
    for i in range(4):
        col_index4.extend(row_index4 + i*config.hidden_size4)

    forward_W_I3 = input_W3[row_index_input34][:, col_index3]
    forward_W_H3 = hidden_W3[row_index3][:, col_index3]

    forward_W_I4 = input_W4[row_index_input34][:, col_index4]
    forward_W_H4 = hidden_W4[row_index4][:, col_index4]

    new_weight['lstm3.weight_ih_l0'] = forward_W_I3.T.contiguous()
    new_weight['lstm3.weight_hh_l0'] = forward_W_H3.T.contiguous()
    new_weight['lstm4.weight_ih_l0'] = forward_W_I4.T.contiguous()
    new_weight['lstm4.weight_hh_l0'] = forward_W_H4.T.contiguous()

    for n, m in model.lstm3.named_parameters():
        if 'bias' in n:
            new_weight['lstm3.'+n] = m.data[col_index3].contiguous()
    for n, m in model.lstm4.named_parameters():
        if 'bias' in n:
            new_weight['lstm4.'+n] = m.data[col_index4].contiguous()

    row_index_input_cls = np.append(row_index3, row_index4 + model.hidden_dim3)
    
    cls = model.softmax.weight.T
    cls = cls[row_index_input_cls]
    new_weight['softmax.weight'] = cls.T.contiguous()
    
    config.hidden_size1 = len(row_index1)
    config.hidden_size2 =  len(row_index2)
    config.hidden_size3 = len(row_index3)
    config.hidden_size4 = len(row_index4)

    print('\nNEW DIMS')
    print('hidden_size1', len(row_index1))
    print('hidden_size2', len(row_index2))
    print('hidden_size3', len(row_index3))
    print('hidden_size4', len(row_index4))
    return new_weight, config


def plot_tensor(t, title):
    t = - (t != 0).astype(int)
    weight_scope = abs(t).max()


    plt.imshow(t.reshape((t.shape[0], -1)),
               vmin=-weight_scope,
               vmax=weight_scope,
               cmap=plt.get_cmap('bwr'),
               interpolation='none')
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    plt.title(title)
    plt.savefig(os.path.join(ARGS.restore_path, '%s.png'%title))

        
def extract_model_filename():
    for i in os.listdir(ARGS.restore_path):
        if 'model_distill' in i:
            return i
        
        
def main():
    print('Restore model from', ARGS.restore_path)
    state_dict = torch.load(os.path.join(ARGS.restore_path, extract_model_filename()),
                            map_location='cpu')
    ARGS.hidden_state = state_dict['lstm1.weight_ih_l0'].shape[1]
    
    if 'lstm4.weight_hh_l0' in state_dict:
         model_type = 'stack_LSTM'
    else:
         model_type = 'stack_biLSTM'
    print('MODEL TYPE:', model_type)
    print()
    
    # загрузка модели
    if model_type == 'stack_biLSTM':
        config = DistilModelConfigBiLSTM()
        config.vocab_size, config.embedding_dim = state_dict['emb.weight'].shape
        model = StackBiLSTM(config, 'cpu')
    elif model_type == 'stack_LSTM':
        config = DistilModelConfigLSTM()
        config.vocab_size, config.embedding_dim = state_dict['emb.weight'].shape
        model = StackLSTM(config, 'cpu')
        
    model.load_state_dict(state_dict)
    model.eval()
    
    if model_type == 'stack_biLSTM':
        matrix = extraxt_matrix_biLSTM(model)
        if ARGS.show_weight == True:
            print('SHOW')
            plot_tensor(matrix['lstm1'][0], 'BiLSTM1')
            plot_tensor(matrix['lstm2'][0], 'BiLSTM2')
            
        if ARGS.cut_model == True:  
            print('CUT')
            coupled_iss1 = extract_iss(matrix['lstm1'][0], matrix['lstm1'][1], 'BiLSTM1', model_type, split_num=2)
            coupled_iss2 = extract_iss(matrix['lstm2'][0], matrix['lstm2'][1], 'BiLSTM2', model_type, split_num=3)
            
            new_weight, config_cut = restructur_biLSTM(model, matrix, config, [coupled_iss1, coupled_iss2])

            model_cut = StackBiLSTM(config_cut, 'cpu')
            model_cut.eval()
        
    elif model_type == 'stack_LSTM':
        h = ARGS.hidden_state
        matrix = extraxt_matrix_LSTM(model)
        if ARGS.show_weight == True:
            print('SHOW')
            plot_tensor(matrix['lstm1'][0], 'LSTM1 forward')
            plot_tensor(matrix['lstm2'][0], 'LSTM1 reverse')
            plot_tensor(matrix['lstm3'][0], 'LSTM2 forward')
            plot_tensor(matrix['lstm4'][0], 'LSTM2 reverse')
            
        if ARGS.cut_model == True:   
            print('CUT')
            coupled_iss1 = extract_iss(matrix['lstm1'][0], matrix['lstm1'][1], 'LSTM1 forward', model_type, split_num=h)
            coupled_iss2 = extract_iss(matrix['lstm2'][0], matrix['lstm2'][1], 'LSTM1 reverse', model_type, split_num=h)
            coupled_iss3 = extract_iss(matrix['lstm3'][0], matrix['lstm3'][1], 'LSTM2 forward', model_type, split_num=2*h)
            coupled_iss4 = extract_iss(matrix['lstm4'][0], matrix['lstm4'][1], 'LSTM2 reverse', model_type, split_num=2*h)

            new_weight, config_cut = restructur_LSTM(model, config, [coupled_iss1, coupled_iss2, coupled_iss3, coupled_iss4])

            model_cut = StackLSTM(config_cut, 'cpu')
            model_cut.eval()
    
    if ARGS.cut_model == True:   
        for n, m in model_cut.named_parameters():
            if n in new_weight:
                m.data = new_weight[n]
            else:
                m.data =  model.state_dict()[n]
        torch.save(model_cut.state_dict(), os.path.join(ARGS.restore_path ,'cut_model_distill_%s.pt'%model_type))

    
if __name__ == "__main__":
    model = main()
            