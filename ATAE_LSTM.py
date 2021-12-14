import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class ATAE_LSTM(nn.Module):
    '''
    word_embedding_dim is the dimension of the word embedding
    input_dim (word_embedding_dim*2) is the dimension of the input which appending the aspect embedding to the word embedding
    hidden_dim is the dimention of the h vector
    '''
    def __init__(self, word_embedding_dim, hidden_dim, batch_size, sentence_length, num_class):
        super().__init__()
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.hidden_plus_aspect_dim = self.hidden_dim + self.word_embedding_dim
        self.input_dim = self.word_embedding_dim * 2
        self.batch_size = batch_size

        self.sentence_length = sentence_length
        self.num_class = num_class

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, batch_first=True) 
        
        # W_HV * K
        self.w_hv = nn.Linear(self.hidden_plus_aspect_dim, self.hidden_plus_aspect_dim)

        # M=tanh(K)
        # W_M * M
        self.w_m = nn.Linear(self.hidden_plus_aspect_dim, self.hidden_dim)

        # alpha = Softmax(W_M * M)
        # r = H alpha
        # W_R * r
        self.w_r = nn.Linear(self.sentence_length, self.hidden_dim)
        
        # W_HN * h_n
        self.w_hn = nn.Linear(self.hidden_dim, self.hidden_dim)

        # W * h*
        self.w_h_star = nn.Linear(self.hidden_dim, self.num_class)
    

    def forward(self, x, prev_hidden):
        H, context = self.lstm(x, prev_hidden)
        h_n = H[:, -1]

        K = self.__get_K(H, x)
        W_H_A = self.w_hv(K)
        M = torch.tanh(W_H_A)
        W_M = self.w_m(M)
        alpha = self.__softmax(W_M, dim=1)

        r = torch.matmul(H, torch.transpose(alpha, 1, 2))

        W_R = self.w_r(r)
        W_HN = self.w_hn(h_n)

        sum = torch.clone(W_R)
        for i, a_batch in enumerate(sum):
            sum[i] = W_R[i] + W_HN[i]

        h_star = torch.tanh(sum)
        W_H_STAR = self.w_h_star(h_star)
        y = self.__softmax(W_H_STAR, dim=1)

        y = y[:, -1]
        return y, H


    def __get_aspect(self, x):
        aspect = x[:, 0, self.word_embedding_dim:]
        return aspect

    def __get_K(self, H, x):
        aspect = self.__get_aspect(x)
        K = torch.clone(H)

        K_list = []
        for i, sentence in enumerate(K):
            aspect_tensor = torch.stack([aspect[i] for _ in range(sentence.shape[0])])
            K_list.append(torch.cat((sentence, aspect_tensor), dim=1))
        K = torch.stack(K_list)
        return K

    def __softmax(self, M, dim):
        softmax_list = []
        for i, m in enumerate(M):
            softmax = F.softmax(m, dim=dim)
            softmax_list.append(softmax)
        softmax_tensor = torch.stack(softmax_list)
        return softmax_tensor

    def init_prev_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim), torch.zeros(1, self.batch_size, self.hidden_dim))
            
