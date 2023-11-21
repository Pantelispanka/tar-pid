import unittest
from tidp.dataloader import MolecularDataset
from tidp.encoder import TipdEncoder
from tidp.decoder import TipdDecoder, TipdDecoder2
from tidp.tokenizer import Tokenizer
from tidp.dyn_tokenizer import DynamicTokenizer
from torch.utils.data import DataLoader
from tidp.tidp import TIPD


# tokenizer = DynamicTokenizer("/Users/pantelispanka/Euclia/tidp/a-syn/a-syn-1000/data/all.csv", col="Docking_molecule")

tokenizer = DynamicTokenizer("/home/pantelispanka/tipd/chembl_500k_train.csv", col="SMILES")

tokenizer.get_smiles()
tokenizer.create_vocab()
tokenizer.create_t2i()
tokenizer.create_i2t()


encoder = TipdEncoder(tokenizer=tokenizer, latent_size=220)
        # decoder = TipdDecoder(50, 120)
decoder = TipdDecoder2(latent_input_size=220, num_channels=292, hidden_size=501, tokenizer=tokenizer)

batch_size = 4
# dataloader = MolecularDataset("/home/pantelispanka/tipd/chembl_500k_train.csv")
dataloader = MolecularDataset("/home/pantelispanka/tipd/chembl_500k_train.csv", column="SMILES")
# dataloader = MolecularDataset("/home/pantelispanka/tipd/all_2.csv", column="SMILES")
train_dataloader = DataLoader(dataloader, batch_size=240, shuffle=True)
tipd = TIPD(encoder=encoder, decoder=decoder, dataloader=train_dataloader, epochs=228, tokenizer=tokenizer)
tipd.train_tipd()
