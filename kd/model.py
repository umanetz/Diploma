import torch
from torch import nn
import torch.nn.functional as F

class StackBiLSTM(nn.Module):
    """2х слойная BiLSTM, в которой прямой и обратный проход рассматриваются как одно целое. 
    Представлена в виде стека двух BiLSTM"""
    def __init__(self, config, device):
        super(StackBiLSTM, self).__init__()
        config.embedding_dim = config.embedding_dim

        self.hidden_dim1 = config.hidden_size1
        self.input_dim2 = config.hidden_size2
        self.num_layers = 1
        self.init_scale = config.init_scale
        self.emb = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.device = device
        self.lstm1 = nn.LSTM(config.embedding_dim, config.hidden_size1, 
                    num_layers=1, batch_first=True, bidirectional=config.bi,
                    dropout=1-config.keep_prob)
        
        self.lstm2 = nn.LSTM(config.hidden_size1*2, config.hidden_size2, 
                    num_layers=1, batch_first=True, bidirectional=config.bi)
        
        self.softmax = nn.Linear(config.hidden_size2*2, config.output_size)
        self.dropout = nn.Dropout(p=1-config.keep_prob)

    def forward(self, input):
        bs, seq_len = input.shape
        self.hidden1 = self.init_hidden(bs, self.hidden_dim1)
        self.hidden2 = self.init_hidden(bs, self.input_dim2)

        xhat = self.emb(input)       
        x = self.dropout(xhat)

        x, self.hidden1 = self.lstm1(x, self.hidden1)  
        x, self.hidden2 = self.lstm2(x, self.hidden2)
        h2, c2 = self.hidden2
        x_lstm = torch.cat((h2[-2], h2[-1]), 1)
        x = self.softmax(x_lstm)
        return x

    def init_hidden(self, batch_size, dim):
          h1 = torch.FloatTensor(self.num_layers*2, batch_size, dim)
          return ((h1.uniform_(-self.init_scale, self.init_scale).to(self.device),
                    h1.uniform_(-self.init_scale, self.init_scale).to(self.device)))
    
    
class StackLSTM(nn.Module):
    """2х слойная BiLSTM, в которой прямой и обратный проход рассматриваются как стек независимых LSTM"""
    def __init__(self, config, device):
        super(StackLSTM, self).__init__()
        config.embedding_dim = config.embedding_dim

        self.hidden_dim1 = config.hidden_size1 # прямая LSTM 1ого слоя
        self.hidden_dim2 = config.hidden_size2 # обратная LSTM 1ого слоя
        self.hidden_dim3 = config.hidden_size3 # прямая LSTM 2ого слоя
        self.hidden_dim4 = config.hidden_size4 # обратная LSTM 2ого слоя
        self.num_layers = 1
        self.init_scale = config.init_scale
        self.device = device
        
        self.emb = nn.Embedding(config.vocab_size, config.embedding_dim)

        self.lstm1 = nn.LSTM(config.embedding_dim, config.hidden_size1, 
                    num_layers=1, batch_first=True, dropout=1-config.keep_prob)
        
        self.lstm2 = nn.LSTM(config.embedding_dim, config.hidden_size2, 
                    num_layers=1, batch_first=True, dropout=1-config.keep_prob)
        
        self.lstm3 = nn.LSTM(config.hidden_size2+config.hidden_size1, config.hidden_size3, 
                    num_layers=1, batch_first=True, dropout=1-config.keep_prob)
        
        self.lstm4 = nn.LSTM(config.hidden_size2+config.hidden_size1, config.hidden_size4, 
                    num_layers=1, batch_first=True, dropout=1-config.keep_prob)
        
        self.softmax = nn.Linear(config.hidden_size3+config.hidden_size4, config.output_size)
        self.dropout = nn.Dropout(p=1-config.keep_prob)

    def forward(self, input):
        bs, seq_len = input.shape
        self.hidden1 = self.init_hidden(bs, self.hidden_dim1)
        self.hidden2 = self.init_hidden(bs, self.hidden_dim2)
        self.hidden3 = self.init_hidden(bs, self.hidden_dim3)
        self.hidden4 = self.init_hidden(bs, self.hidden_dim4)

        xhat = self.emb(input)       
        x = self.dropout(xhat)
        x_revers = x.flip([1]) # переворачиваем последовательнось
        x1, self.hidden1 = self.lstm1(x, self.hidden1)  
        x2, self.hidden2 = self.lstm2(x_revers, self.hidden2)

        x = torch.cat((x1, x2), -1) # конкатенация прямого и обратного прохода
        x_revers = x.flip([1]) # переворачиваем последовательнось

        x3, self.hidden3 = self.lstm3(x, self.hidden3)  
        x4, self.hidden4 = self.lstm4(x_revers, self.hidden4)  
        x = torch.cat((x3[:,-1], x4[:,-1]), -1) # конкатенация прямого и обратного прохода
        x = self.softmax(x)
        return x

    def init_hidden(self, batch_size, dim):
          h1 = torch.FloatTensor(self.num_layers, batch_size, dim)
          return (h1.uniform_(-self.init_scale, self.init_scale).to(self.device),
                    h1.uniform_(-self.init_scale, self.init_scale).to(self.device))
    