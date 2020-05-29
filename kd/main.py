import argparse

import time
import json
import numpy as np
from time import gmtime, strftime
import torch
from torch import nn
from torch import optim
import argparse

import time
import json
import numpy as np
from time import gmtime, strftime
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from  torch.utils.data import DataLoader

from datareader import raw_data
from model import StackBiLSTM, StackLSTM
from tqdm import tqdm

from sklearn.metrics import f1_score
import os.path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help='Путь к папкам с данными: tokens/ и logits/')
    parser.add_argument("--model_type", help='Тип модели: stack_biLSTM, stack_LSTM')
    parser.add_argument("--regularizer", default="group_lasso", help='Тип регуляризации: l1_regularizer, group_lasso')
    parser.add_argument("--save_path", default="./", help='Путь, куда будет сохраняться модель')
    parser.add_argument("--max_epoch", default=20, type=int, help='Кол-во эпох обучения')
    parser.add_argument("--hidden_size", default=1000, type=int, help='размер скрытого слоя')
    parser.add_argument("--restore_model_path",  default=None, help='название файла с моделью')
    parser.add_argument("--use_cuda", action='store_true', default=True)

    args = parser.parse_args()
    args.device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

    return args


ARGS = get_args()
ARGS.zero_threshold = 0.0001


class Grouplasso:
    """Считаем значение групповой регуляризации для моделей двух типов"""
    def __init__(self, model_type):
        self.model_type = model_type
        
    def add_structure_grouplasso(self, var, coupled_var, split_num=2):
        t1 = torch.pow(var, 2)

        t1_col_sum = torch.sum(t1, dim=0)
        t1_row_sum = torch.sum(t1, dim=1)
        
        if self.model_type == 'stack_biLSTM':
            t1_col_sum = torch.chunk(t1_col_sum, 4*2) 
            t1_row_sum2 = torch.chunk(t1_row_sum, split_num)[-1]

            t2 = torch.pow(coupled_var, 2)
            t2_row_sum = torch.sum(t2, dim=1)
            t2_row_sums = torch.chunk(t2_row_sum, 2)   

            reg_sum = t1_row_sum2 + \
                      t2_row_sums[0] + t2_row_sums[1]+ 1.0e-8
            
        elif self.model_type == 'stack_LSTM':
            t1_col_sum = torch.chunk(t1_col_sum, 4) 
            t1_row_sum2 = t1_row_sum[-split_num:]

            t2 = torch.pow(coupled_var, 2)
            t2_row_sum = torch.sum(t2, dim=1)

            reg_sum = t1_row_sum2 + t2_row_sum + 1.0e-8
        
        for t in t1_col_sum:
             reg_sum = reg_sum + t

        reg_sqrt = torch.sqrt(reg_sum)
        reg = torch.sum(reg_sqrt)
        return reg   

    
    def bilstm1_reg(self, model):
        '''структурные компоненты первого слоя для модели типа stack_BiLSTM'''
        var = torch.cat([model.lstm1.weight_ih_l0.T, model.lstm1.weight_hh_l0.T]) 
        var_revers = torch.cat([model.lstm1.weight_ih_l0_reverse.T, model.lstm1.weight_hh_l0_reverse.T])
        var = torch.cat([var, var_revers], 1) # (2*hs, 8*hs)
        
        couple_var = torch.cat([model.lstm2.weight_ih_l0.T, model.lstm2.weight_ih_l0_reverse.T], 1) # (2*hs, 8*hs)
        return self.add_structure_grouplasso(var, couple_var, split_num=2)

    def bilstm2_reg(self, model):
        '''структурные компоненты второго слоя для модели типа stack_BiLSTM'''
        var = torch.cat([model.lstm2.weight_ih_l0.T, model.lstm2.weight_hh_l0.T]) 
        var_revers = torch.cat([model.lstm2.weight_ih_l0_reverse.T, model.lstm2.weight_hh_l0_reverse.T])
        var = torch.cat([var, var_revers], 1) # (2*hs, 8*hs)
        couple_var = model.softmax.weight.T # (2*hs, output_dim)
        
        return self.add_structure_grouplasso(var, couple_var, split_num=3)
    
    def lstm1_reg(self, model):
        '''структурные компоненты прямого прохода первого слоя для модели типа stack_LSTM'''
        hs = model.hidden_dim1
        var = torch.cat([model.lstm1.weight_ih_l0.T, model.lstm1.weight_hh_l0.T]) # (2*hs, 4*hs)
        couple_var = torch.cat([model.lstm3.weight_ih_l0.T[:hs], model.lstm4.weight_ih_l0.T[:hs]], -1) # (hs, 8*hs)
        return self.add_structure_grouplasso(var, couple_var, split_num=hs)

    def lstm2_reg(self, model):
        '''структурные компоненты обратного прохода первого слоя для модели типа stack_LSTM'''
        hs = model.hidden_dim2
        var = torch.cat([model.lstm2.weight_ih_l0.T, model.lstm2.weight_hh_l0.T]) # (2*hs, 4*hs)
        couple_var = torch.cat([model.lstm3.weight_ih_l0.T[-hs:], model.lstm4.weight_ih_l0.T[-hs:]], -1) # (hs, 8*hs)
        return self.add_structure_grouplasso(var, couple_var, split_num=hs)

    def lstm3_reg(self, model):
        '''структурные компоненты прямого прохода второго слоя для модели типа stack_LSTM'''
        hs = model.hidden_dim3
        var = torch.cat([model.lstm3.weight_ih_l0.T, model.lstm3.weight_hh_l0.T]) # (2*hs, 4*hs)
        couple_var = model.softmax.weight.T[:hs] # (hs, output_dim)
        return self.add_structure_grouplasso(var, couple_var, split_num=hs)

    def lstm4_reg(self, model):
        '''структурные компоненты обратного прохода второго слоя для модели типа stack_LSTM'''
        hs = model.hidden_dim4
        var = torch.cat([model.lstm4.weight_ih_l0.T, model.lstm4.weight_hh_l0.T]) # (2*hs, 4*hs)
        couple_var = model.softmax.weight.T[-hs:] # (hs, output_dim)
        return self.add_structure_grouplasso(var, couple_var, split_num=hs) 

    def get_coef(self, model, coef):
        # считаем значение групповой регуляризации
        if self.model_type == 'stack_biLSTM':
            return (self.bilstm1_reg(model) + self.bilstm2_reg(model))* coef
        if self.model_type == 'stack_LSTM':
            return (self.lstm1_reg(model) + self.lstm3_reg(model) + self.lstm4_reg(model)) * coef +\
        self.lstm2_reg(model) * 0.001
        

def get_regularization(model, config, lasso_class):
    '''Функция предназначена для подсчета L1 и GroupLasso регуляризаций'''
    
    _regularization = config.weight_decay # коэфициент L1 регуляризации
    l1_regularization = 0
    
    for n, m in  model.named_parameters():
        if 'emb' not in n:
            l1_regularization += m.abs().sum()
    _regularization = _regularization * l1_regularization   

    coef = config.group_decay # коэфициент GroupLasso регуляризации
    if coef > 0:
        _regularization = _regularization + lasso_class.get_coef(model, coef)
    return _regularization


def zero_weght(model, config):
    '''Обнуляем веса, которые ниже порога zero_threshold'''
    zero_threshold = ARGS.zero_threshold
    sparsity = {}

    if config.weight_decay > 0 or config.group_decay > 0:
        threshold = max(zero_threshold, 2*config.weight_decay)
        for sp_name, m in model.named_parameters():
            if 'emb' in sp_name:
                continue
            where_cond = torch.abs(m) < threshold
            m2 = m.data
            m2[where_cond] = 0.0
            m.data = m2
            if 'weight' in sp_name:
                s = torch.mean((m2==0).float())
                sparsity[sp_name + '_elt_sparsity'] = s.item()

    return sparsity 


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
class DistilModelConfigBiLSTM(object):
    """Конфигурация для модели stack_biLSTM"""
    def __init__(self):
        self.init_scale = 0.04
        self.max_grad_norm = 4
        self.hidden_size1 = ARGS.hidden_size
        self.hidden_size2 = ARGS.hidden_size
        self.keep_prob = 0.60
        self.batch_size = 128
        self.output_size = 80
        self.bi = True
        self.weight_decay = 1e-5 # коэффициент l1 регуляризации
        self.group_decay = 1e-3 # коэффициент GroupLasso регуляризации
        self.learning_rate = 1
        self.embedding_dim = ARGS.hidden_size
        self.vocab_size = 32530
        self.alpha = 0.9  # степень симмуляции поведения модели учителя 
        self.T = 4.0 # параметр температуры 
        
        
class DistilModelConfigLSTM(object):
    """Конфигурация для модели stack_LSTM"""
    def __init__(self):
        self.init_scale = 0.04
        self.max_grad_norm = 4
        self.hidden_size1 = ARGS.hidden_size
        self.hidden_size2 = ARGS.hidden_size
        self.hidden_size3 = ARGS.hidden_size
        self.hidden_size4 = ARGS.hidden_size
        self.max_epoch = 20
        self.keep_prob = 0.60
        self.batch_size = 128
        self.output_size = 80
        self.bi = True
        self.weight_decay = 1e-5
        self.group_decay = 1e-3
        self.learning_rate = 1
        self.embedding_dim = ARGS.hidden_size
        self.vocab_size = 32530
        self.alpha = 0.9
        self.T = 4.0

        
def f1_metrics(pred, true):
    pred = pred.argmax(1).data.numpy()
    true = true.data.numpy()
    return f1_score(true, pred, average='macro')


def distillation(y, teacher_scores, labels, T, alpha):
    '''loss функция для Knowledge distillation'''
    p = F.log_softmax(y.float()/T, dim=1)
    q = F.softmax(teacher_scores.float()/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_ce * (1. - alpha) + l_kl * alpha


def train_distill(model, optimizer, criterion, dataloader, config, grouplasso_):
    outputs = {}
    costs = 0.0
    regularizations = 0
    iters = 0
    f1 = 0

    model.train()
    for batch in tqdm(dataloader, desc='train...'):
        batch = {t: batch[t].to(ARGS.device) for t in batch}
        bert_output = batch['bert_prob']
        optimizer.zero_grad()

        output = model(batch['tokens'])
        l1_coef = get_regularization(model, config, grouplasso_)
        sparsity = zero_weght(model, config)
        loss = criterion(output, bert_output, batch['target'], config.T, config.alpha)
        loss = loss + l1_coef
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        f1 += f1_metrics(output.cpu(), batch['target'].cpu())
        costs += loss.item()
        
    outputs['cross_entropy'] = costs / len(dataloader)
    outputs['f1'] = f1 / len(dataloader)
    return outputs


def test_distill(model, criterion, dataloader, config, grouplasso_):
    sparsity = zero_weght(model, config)
    outs = []
    costs = 0
    targets = []
    model.eval()
    outputs = {}
    l1_coef = get_regularization(model, config, grouplasso_) 
    
    for batch in dataloader:
        batch = {t: batch[t].to(ARGS.device) for t in batch} 
        bert_output = batch['bert_prob']
        output = model(batch['tokens'])
        costs += (criterion(output, bert_output, batch['target'], config.T, config.alpha) + l1_coef).item()
        outs.extend(output.cpu().data.numpy())
        targets.extend(batch['target'].cpu())
    outputs['f1'] = f1_score(np.array(targets), np.array(outs).argmax(-1), average='macro')
    outputs['cross_entropy'] = costs / len(dataloader)
    outputs['sparsity'] = sparsity
    return outputs

def restore_model(model_path):
    '''Загружаем ранее сохраненную модель. model_path - путь к модели'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state_dict = torch.load(model_path, map_location=device)
    
    if ARGS.model_type == 'stack_biLSTM':
        config = DistilModelConfigBiLSTM()
        config.vocab_size, config.embedding_dim = model_state_dict['emb.weight'].shape
        config.hidden_size1 = model_state_dict['lstm1.weight_hh_l0'].shape[-1]
        config.hidden_size2 = model_state_dict['lstm2.weight_hh_l0'].shape[-1]
        config.output_size = model_state_dict['softmax.weight'].shape[0]
        model = StackBiLSTM(config, device).to(device)
    elif ARGS.model_type == 'stack_LSTM':
        config = DistilModelConfigLSTM()
        config.vocab_size, config.embedding_dim = model_state_dict['emb.weight'].shape
        config.hidden_size1 = model_state_dict['lstm1.weight_hh_l0'].shape[-1]
        config.hidden_size2 = model_state_dict['lstm2.weight_hh_l0'].shape[-1]
        config.hidden_size3 = model_state_dict['lstm3.weight_hh_l0'].shape[-1]
        config.hidden_size4 = model_state_dict['lstm4.weight_hh_l0'].shape[-1]
        config.output_size = model_state_dict['softmax.weight'].shape[0]
        model = StackLSTM(config, device).to(device)
    
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


def main():
    print('MODEL TYPE:', ARGS.model_type)
    if ARGS.model_type == 'stack_biLSTM':
        config = DistilModelConfigBiLSTM()
    elif ARGS.model_type == 'stack_LSTM':
        config = DistilModelConfigLSTM()
        
    max_epoch = ARGS.max_epoch
    
    # параметры обучения
    if ARGS.regularizer == 'l1_regularizer':
        config.group_decay = 0
        step_size = 5
        
    elif ARGS.regularizer == 'group_lasso':
        config.weight_decay = 0
        step_size = 9

    train_set, valid_set, test_set = raw_data(ARGS.data_path, load_probs=True)
    train_loader = DataLoader(train_set, config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, config.batch_size, shuffle=False)
    
    if ARGS.model_type == 'stack_biLSTM':
        model = StackBiLSTM(config, ARGS.device).to(ARGS.device)
    elif ARGS.model_type == 'stack_LSTM':
        model = StackLSTM(config, ARGS.device).to(ARGS.device)

    optimizer = optim.SGD(model.parameters(), lr=1.0)
    criterion = distillation
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    
    subfolder_name = strftime("%Y-%m-%d___%H-%M-%S", gmtime())
    save_path = os.path.join(ARGS.save_path, subfolder_name)
    grouplasso = Grouplasso(ARGS.model_type)
    
    if ARGS.restore_model_path:
        print('Restore model from', ARGS.restore_model_path)
        model = restore_model(ARGS.restore_model_path)
        test_metics = test_distill(model, criterion, test_loader, config, grouplasso)
        print(f"Test Loss: {test_metics['cross_entropy']:.3f}, Test F1: {test_metics['f1']:.3f}")
        return 

    else:
        os.mkdir(save_path)
        print('Save model at', save_path)

    train_loss = []
    test_loss = []
    best_f1 = 0

    for epoch in range(max_epoch):

        train_metics = train_distill(model, optimizer, criterion, train_loader, config, grouplasso)
        train_loss.append(train_metics)

        test_metics = test_distill(model, criterion, valid_loader, config, grouplasso)
        test_loss.append(test_metics)
#         clear_output(wait=False)

        print(f"Epoch: {epoch+1:02}, Train Loss: {train_metics['cross_entropy']:.3f},  Train F1: {train_metics['f1']:.3f}")
        print(f"\tEval Loss: {test_metics['cross_entropy']:.3f}, Eval F1: {test_metics['f1']:.3f}")
        print()
        get_lr(optimizer)
        scheduler.step()
              
#         for s in test_metics['sparsity']:
#           if 'bias' in s:
#               continue
#           print(s, test_metics['sparsity'][s])
#           print()
              
        if best_f1 < test_metics['f1']:
            best_f1 = test_metics['f1']
            torch.save(model.state_dict(), os.path.join(save_path , 'model_distill_%s.pt'%ARGS.model_type))
              
        np.save(os.path.join(save_path, 'dev_metrics.npy'), test_loss)
              
    test_metics = test_distill(model, criterion, test_loader, config, grouplasso)
    print(f"Test Loss: {test_metics['cross_entropy']:.3f}, Test F1: {test_metics['f1']:.3f}")
          
    return model
  
          
if __name__ == "__main__":
    model = main()
        
      
#!python main.py --data_path ./files/data/preprossessing_data/ --regularizer group_lasso
