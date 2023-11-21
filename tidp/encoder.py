import torch
import torch.nn.functional as F
from torch import nn
from tidp.tokenizer import Tokenizer


class TipdEncoder(nn.Module):
    def __init__(self, tokenizer: Tokenizer, hidden_size=256, num_layers=2, latent_size=50,
                 bidirectional=False):
        super(TipdEncoder, self).__init__()
        self.tokenizer = tokenizer
        self.embs = nn.Embedding(self.tokenizer.get_vocab_size(), hidden_size)
        self.rnn = nn.LSTM(input_size=hidden_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional)

        self.final_mlp_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, latent_size))

        self.final_mlp_2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, latent_size))

        self.conv_1 = nn.Conv1d(tokenizer.mol_length, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.linear_0 = nn.Linear(250, 435)
        self.linear_1 = nn.Linear(435, latent_size)
        self.linear_2 = nn.Linear(435, latent_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    def encode(self, sm_list):
        """
        Maps smiles onto a latent space
        """

        tokens, lens = self.tokenizer.encode(sm_list)
        x = F.one_hot(tokens, num_classes=self.tokenizer.get_vocab_size())
        # to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)
        #
        # outputs = self.rnn(self.embs(to_feed))[0]
        # outputs = outputs[lens, torch.arange(len(lens))]
        #
        # return self.final_mlp_1(outputs), self.final_mlp_2(outputs)
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)


    def forward(self, sm_list):
        """
        Maps smiles onto a latent space
        """
        # tokens, lens = self.tokenizer.encode(sm_list)
        tokens = self.tokenizer.encode(sm_list)
        # print(tokens.size())
        toks = tokens.long()
        
        toks = toks.to(torch.device("cuda:0"))

        x = F.one_hot(toks, num_classes=self.tokenizer.get_vocab_size())

        x = x.float()
        # print(x.size())
        # print(x)
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)


        # print(tokens)
        # to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)

        # outputs = self.rnn(self.embs(to_feed))[0]
        # outputs = outputs[lens, torch.arange(len(lens))]

        return self.final_mlp_1(outputs), self.final_mlp_2(outputs)
