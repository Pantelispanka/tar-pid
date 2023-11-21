import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from math import pi, log
warnings.filterwarnings("ignore")

class TIPD(nn.Module):

    def __init__(self, encoder, decoder, dataloader, epochs, tokenizer, loss=None):
        super(TIPD, self).__init__()
        self.device = torch.device("cuda:0")
        
        self.enc = encoder.to(self.device)
        self.dec = decoder.to(self.device)
        self.data_loader = dataloader
        self.epochs = epochs
        self.loss = loss
        self.tokenizer = tokenizer
        self.latent_descr = 50
        # self.model = MolecularVAE()
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.003)


    def train_tipd(self):
        train_loss = 0
        print("Starting training for epochs: {epochs}".format(epochs=self.epochs))
        for i in range(self.epochs):
            print("Epoch {0}".format(i))
            total_loss = 0
            for sm, vals in self.data_loader:
                self.optimizer.zero_grad()
                smi_x = self.tokenizer.encode(sm)

                smi_x = smi_x.to(self.device)
                # output, mean, logvar = self.model(sm)


                lats, mean = self.enc(sm)
                s = self.sampling(lats, mean)
                # y = self.dec(s, lats, mean)

                output, mean, logvar = self.dec(s, lats, mean)

                loss = self.vae_loss(output, smi_x, lats, mean)
                loss.backward()
                total_loss += loss
                self.optimizer.step()
                # samples = self.sample(10)
                # print(y[0].size())
                outp = output.cpu().detach().numpy()
                # sampled = outp.reshape(1, 120, self.tokenizer.get_vocab_size()).argmax(axis=2)[0]
                # samples = torch.argmax(y[0], dim=2)
                # reshaped = output.reshape(1, 120, self.tokenizer.get_vocab_size())



                # samples = outp.reshape(1, 120, self.tokenizer.get_vocab_size()).argmax(axis=2)[0]
                samples = torch.argmax(output, dim=2)

                # print(samples)
                smis = self.tokenizer.decode(samples)
                print(sm[:2])
                print(smis[:2])
            print('train', total_loss / len(self.data_loader.dataset))

    def sample(self, num_instances: int = 20):
        samples = self.dec.sample(num_instances)
        return samples

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean
    #
    # def vae_loss(self, x_decoded_mean, x, z_mean, z_logvar):
    #     # means, log_stds = torch.split(self.enc.encode(x),
    #     #                               len(self.latent_descr), dim=1)
    #     xent_loss = F.binary_cross_entropy(x_decoded_mean.float(), x[0].float(), reduction='sum')
    #     kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
    #     return xent_loss + kl_loss

    def vae_loss(self, x_decoded_mean, x, z_mean, z_logvar):
        # print(x_decoded_mean[0].size())
        # print(x[0].size())
        # toks = x.long()
        x = F.one_hot(x.long(), num_classes=self.tokenizer.get_vocab_size())
        xent_loss = F.binary_cross_entropy(x_decoded_mean.float(), x.float(), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return xent_loss + kl_loss

    def get_elbo(self, x, y):
        means, log_stds = torch.split(self.enc.encode(x),
                                      len(self.latent_descr), dim=1)
        latvar_samples = (means + torch.randn_like(log_stds) *
                          torch.exp(0.5 * log_stds))

        rec_part = self.dec.weighted_forward(x, latvar_samples).mean()

        normal_distr_hentropies = (log(2 * pi) + 1 + log_stds).sum(dim=1)

        latent_dim = len(self.latent_descr)
        condition_dim = len(self.feature_descr)

        zy = torch.cat([latvar_samples, y], dim=1)
        log_p_zy = self.lp.log_prob(zy)

        y_to_marg = latent_dim * [True] + condition_dim * [False]
        log_p_y = self.lp.log_prob(zy, marg=y_to_marg)

        z_to_marg = latent_dim * [False] + condition_dim * [True]
        log_p_z = self.lp.log_prob(zy, marg=z_to_marg)
        log_p_z_by_y = log_p_zy - log_p_y
        log_p_y_by_z = log_p_zy - log_p_z

        kldiv_part = (-normal_distr_hentropies - log_p_zy).mean()

        elbo = rec_part - self.beta * kldiv_part
        elbo = elbo + self.gamma * log_p_y_by_z.mean()

        return elbo, {
            'loss': -elbo.detach().cpu().numpy(),
            'rec': rec_part.detach().cpu().numpy(),
            'kl': kldiv_part.detach().cpu().numpy(),
            'log_p_y_by_z': log_p_y_by_z.mean().detach().cpu().numpy(),
            'log_p_z_by_y': log_p_z_by_y.mean().detach().cpu().numpy()
        }
