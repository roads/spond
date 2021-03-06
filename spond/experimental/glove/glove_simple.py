# Test bed for Glove loss


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

#from utils import setup_torch

#device = setup_torch()



class GloveEmbedding(nn.Embedding):
    # Embedding that has an attached glove loss

    def __init__(self, num_embeddings, embedding_dim, co_occurrence,
                 device='cpu', x_max=100, alpha=0.75):
        # co_occurrence must be a matrix of index to incidence
        super(GloveEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.co_occurrence = co_occurrence
        self.device = device
        self.x_max = x_max
        self.alpha = alpha

    def weight_func(self, x):
        # x: co_occurrence values
        wx = (x/self.x_max)**self.alpha
        wx = torch.min(wx, torch.ones_like(wx))
        return wx.to(self.device)

    def wmse_loss(self, indices):
        # inputs are indexes, targets are actual embeddings
        # In the actual algorithm, "inputs" is the weight_func run on the
        # co-occurrence statistics.
        # loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
        # not sure what it should be replaced by here
        # "targets" are the log of the co-occurrence statistics.
        # need to make every pair of indices that exist in co-occurrence file
        used_indices = indices
        targets = self.co_occurrence[used_indices]
        weights_x = self.weight_func(targets)
        loss = weights_x * F.mse_loss(self.inputs, torch.log(targets), reduction='none')
        return torch.mean(loss).to(self.device)


class GloveSimple(pl.LightningModule):

    def __init__(self, train_embeddings_file, batch_size, limit=None,
                 train_cooccurrence_file=None):
        # train_enbeddings_file: the filename contaning the pre-trained weights
        # train_cooccurrence_file: the filename containing the co-occurrence statistics
        # that we want to match these embeddings to. If passed, a GloveEmbedding layer
        # will be used.
        super(GloveSimple, self).__init__()
        self.limit = limit
        self.train_data = torch.load(train_embeddings_file)
        self.train_cooccurrence = train_cooccurrence_file
        if self.train_cooccurrence:
            self.train_cooccurrence = torch.load(self.train_cooccurrence)
        self.batch_size = batch_size
        nemb, dim = self.train_data['wi.weight'].shape
        self.pretrained_embeddings = (
            self.train_data['wi.weight'] +
            self.train_data['wj.weight']
        )
        if limit:
            nemb = limit
        self.num_embeddings = nemb
        self.embedding_dim = dim
        if not self.train_cooccurrence:
            self.emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        else:
            self.emb_layer = GloveEmbedding(self.num_embeddings, self.embedding_dim,
                                            self.train_cooccurrence, device=self.device)
        self.emb_layer.weight.data.uniform_(-1, 1)
        # get rid of this
        #self.linear_layer = nn.Linear(self.num_embeddings, self.embedding_dim)
        # TODO:
        # What is the intended output?
        # Should it be that we feed in an index, and we get an embedding
        # or is it that we look at the emb_layer.weight values?

        #self.layer = nn.Sequential(
        #    nn.Embedding(self.num_embeddings, self.embedding_dim),
        #    nn.Linear(self.num_embeddings, self.embedding_dim)
        #)

    def forward(self, indices):
        set_indices = (np.arange(len(indices)), indices)

        batch = torch.zeros((len(indices), self.num_embeddings), dtype=int).to(self.device)

        batch[set_indices] = 1
        return self.emb_layer(batch)

    def training_step(self, batch, batch_idx):
        #opt = self.optimizers()
        # input: indices of embedding
        # targets: target embeddings
        indices, targets = batch
        out = self.emb_layer(indices)
        # the loss should be between the targets and the
        # embeddings for the items in this batch.
        loss = F.mse_loss(out, targets)

        #self.manual_backward(loss, opt, retain_graph=True)
        #self.manual_backward(loss, opt)
        #opt.step()
        #opt.zero_grad()
        print(f"loss: {loss}")
        # 1. learn itself
        # 2. optimise 2 separate loss objectives - one is GloVe loss,
        #    which we will inject similar to regularisation mechanism
        #    then w.r.t second loss to be specified by the user
        # 3. We want a specific layer type

        return loss

    def configure_optimizers(self):
        #opt = optim.Adagrad(self.parameters(), lr=0.05)
        opt = optim.Adam(self.parameters(), lr=0.005)
        return opt

    def train_dataloader(self):
        dataset = GloveEmbeddingsDataset(self.train_data, self.limit, self.device)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class GloveEmbeddingsDataset(Dataset):
    # Dataset for existing embedings

    def __init__(self, data, limit=None, device='cpu'):
        # train_data contains wi.weight / wj.weight / bi.weight / bj.weight
        # for stability, the target is the wi + wj
        self.weights = data['wi.weight'] + data['wj.weight']
        if limit:
            self.weights = self.weights[:limit]
        nemb, dim = self.weights.shape
        self.device = device
        self.x = torch.arange(nemb).to(self.device)
        self.y = self.weights.to(self.device)
        self.N = nemb

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.x[idx]
        y = self.y[idx]
        return x, y


if __name__ == '__main__':

    model = GloveSimple('glove_audio.pt', batch_size=100)#, limit=1000)
    trainer = pl.Trainer(gpus=0, max_epochs=100, progress_bar_refresh_rate=20)
    trainer.fit(model)
