from torch import nn
import torch
import torch.nn.functional as F

from tidp.tokenizer import Tokenizer


class TipdDecoder(nn.Module):

    def __init__(self, latent_input_size, num_channels
                 , hidden_size: int = 100, constraints: int = 20
                 , num_layers: int = 4
                 , bidirectional: bool = True):
        super(TipdDecoder, self).__init__()
        self.latent_input_size = latent_input_size
        self.latent_conc_shape = latent_input_size + constraints

        self.latent_fc = nn.Linear(self.latent_conc_shape, num_channels)
        # self.latent_constraints = nn.Linear(constraints, num_channels)
        self.rnn = nn.LSTM(input_size=num_channels,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 120))

    def forward(self, latent_vars, constraints):
        # print("----INPUT----")
        # print(latent_vars.size())
        # print(constraints.size())
        # print(latent_vars)
        # print(constraints)
        T = torch.cat((latent_vars, constraints), -1)
        # print("-----CAT-----")
        # print(T.size())
        # print(T)
        y = self.latent_fc(T)
        # c = self.latent_constraints(constraints)
        y = self.rnn(y)
        y = y[0]
        y = self.final_mlp(y)
        # print(y)
        # y = self.final_mlp(y[0])
        # y = F.log_softmax(y)
        # print(y)
        z = y.view(y.size(0), 1, y.size(-1)).repeat(1, 120, 1)
        # y = torch.reshape(y, [])
        z = F.log_softmax(z, dim=1)
        # print(z)
        # print(z.size())

        ans_seqs = [[1] for _ in range(2)]
        ans_logits = []
        argmax = True
        logits = y.detach()
        logits = torch.log_softmax(logits, dim=-1)
        ans_logits.append(logits.unsqueeze(0))

        if argmax:
            cur_tokens = torch.max(z, dim=-1)[1]
        else:
            cur_tokens = torch.multinomial(F.softmax(logits, dim=-1), 1)

        # print(cur_tokens)
        tokenizer = Tokenizer(120)
        sm = tokenizer.decode(cur_tokens)
        # print(cur_tokens)
        print(sm)
        det_tokens = cur_tokens.cpu().detach().tolist()
        # print(det_tokens)
        ans_seqs = [a + b for a, b in zip(ans_seqs, det_tokens)]

        ans_logits = torch.cat(ans_logits, dim=0)
        # ans_seqs = torch.tensor(ans_seqs)[:, 1:]
        # print(ans_logits)
        # print(ans_seqs)
        return cur_tokens

    def sample(self, num_instances, constraints=torch.FloatTensor([-7.0, -7.0, -7.0, -7.0, -7.0
                                                                   , -7.0, -7.0, -7.0, -7.0, -7.0
                                                                   , -7.0, -7.0, -7.0, -7.0, -7.0
                                                                   , -7.0, -7.0, -7.0, -7.0, -7.0])):

        for instance in range(num_instances):
            latents = torch.rand(self.latent_input_size)
            # T = torch.cat((latents, constraints), -1)
            # print(T)
            self.forward(latents, constraints)


class TipdDecoder2(nn.Module):

    def __init__(self, latent_input_size, num_channels
                 , hidden_size: int = 100, constraints: int = 20
                 , num_layers: int = 4
                 , bidirectional: bool = True, tokenizer: Tokenizer = None):
        super(TipdDecoder2, self).__init__()
        self.latent_input_size = latent_input_size
        self.latent_conc_shape = latent_input_size + constraints

        self.tokenizer = tokenizer

        self.linear_3 = nn.Linear(self.latent_input_size, num_channels)
        self.rnn = nn.GRU(num_channels, hidden_size, 3, batch_first=True)
        # self.rnn = nn.LSTM(input_size=num_channels,
        #                    hidden_size=hidden_size,
        #                    num_layers=num_layers,
        #                    bidirectional=bidirectional)
        self.linear_4 = nn.Linear(hidden_size, self.tokenizer.get_vocab_size())

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, latent_vars, z_mean, z_logvar):
        # z_mean, z_logvar = self.encode(x)
        # z = self.sampling(z_mean, z_logvar)
        return self.decode(latent_vars), z_mean, z_logvar

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, self.tokenizer.mol_length, 1)
        output, hn = self.rnn(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = F.softmax(self.linear_4(out_reshape), dim=1)
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        # print("DECODER")
        # print(y.size())
        return y

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def sample(self, num_instances, constraints=torch.FloatTensor([-7.0, -7.0, -7.0, -7.0, -7.0
                                                                   , -7.0, -7.0, -7.0, -7.0, -7.0
                                                                   , -7.0, -7.0, -7.0, -7.0, -7.0
                                                                   , -7.0, -7.0, -7.0, -7.0, -7.0])):
        latents = torch.rand((num_instances, self.latent_input_size))
        return self.decode(latents)
        # for instance in range(num_instances):
        #     latents = torch.rand((num_instances, self.latent_input_size))
        #     # T = torch.cat((latents, constraints), -1)
        #     # print(T)
        #     self.decode(latents)
