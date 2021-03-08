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



class GloveModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, co_occurrence,
                 device='cpu', x_max=100, alpha=0.75):
        super(GloveModule, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

        self.co_occurrence = co_occurrence.coalesce()
        self.device = device
        self.x_max = x_max
        self.alpha = alpha

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()
        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x

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
        from itertools import combinations
        # Not every index will be represented in the co_occurrence matrix
        # To calculate glove loss, we will take all the pairs in the co-occurrence
        # that contain anything in the current set of indices.
        allindices = self.co_occurrence.indices()
        # The following is terrible, but figure it out once we get it working
        # Converting to numpy is very slow
        intersection = set(indices.numpy()).intersection(
            set(allindices.unique().numpy()))
        if not len(intersection):
            return 0
        intersection = np.array(list(intersection))
        allpairs = torch.vstack([
            allindices.t()[allindices[0] == index]
            for index in intersection
        ])
        # cannot index into a sparse tensor with pairs
        # so we have to find the indices of these pairs, and then
        # find the values corresponding to those.
        targets = torch.Tensor([
            self.co_occurrence[item[0]][item[1]]
            for item in allpairs
        ])
        weights_x = self.weight_func(targets)
        i_indices = allpairs[:, 0]
        j_indices = allpairs[:, 1]
        current_weights = self(i_indices, j_indices)
        loss = weights_x * F.mse_loss(
            current_weights, torch.log(targets), reduction='none')
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
        self.emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        if self.train_cooccurrence is not None:
            self.glove_layer = GloveModule(
                self.num_embeddings, self.embedding_dim,
                self.train_cooccurrence, device=self.device)
        else:
            self.glove_layer = None
        self.emb_layer.weight.data.uniform_(-1, 1)


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
        if self.glove_layer is not None:
            loss += self.glove_layer.wmse_loss(indices)
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

    model = GloveSimple('glove_audio.pt', batch_size=100,
                        train_cooccurrence_file='../audioset/co_occurrence_audio_all.pt')#, limit=1000)
    trainer = pl.Trainer(gpus=0, max_epochs=100, progress_bar_refresh_rate=20)
    trainer.fit(model)
