# Test bed for Glove loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class GloveLayer(nn.Embedding):
    # TODO: Is there a way to express constraints on weights other than
    # the nn.Functional.gradient_clipping ?
    def __init__(self, num_embeddings, embedding_dim, co_occurrence,
                 # glove learning options
                 x_max=100, alpha=0.75,
                 # whether or not to use the wi and wj
                 # if set to False, will use only wi
                 double=False,
                 # nn.Embedding options go here
                 padding_idx=None,
                 scale_grad_by_freq=None,
                 max_norm=None, norm_type=2,
                 sparse=False   # not supported- just here to keep interface
                 ):
        # Not calling Embedding constructor, as this module is
        # composed of Embeddings. However we have to set some attributes
        nn.Embedding.__init__(self, num_embeddings, embedding_dim,
                              padding_idx=None, max_norm=None,
                              norm_type=2.0, scale_grad_by_freq=False,
                              sparse=False, _weight=None)
        if sparse:
            raise NotImplementedError("`sparse` is not implemented for this class")
        # for the total weight to have a max norm of K, the embeddings
        # that are summed to make them up need a max norm of K/2
        used_norm = max_norm / 2 if max_norm else None
        kws = {}
        if used_norm:
            kws['max_norm'] = used_norm
            kws['norm_type'] = norm_type
        if padding_idx:
            kws['padding_idx'] = padding_idx
        if scale_grad_by_freq is not None:
            kws['scale_grad_by_freq'] = scale_grad_by_freq
        self.double = double
        self.wi = nn.Embedding(num_embeddings, embedding_dim, **kws)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.wi.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()

        if self.double:
            self.wj = nn.Embedding(num_embeddings, embedding_dim, **kws)
            self.bj = nn.Embedding(num_embeddings, 1)

            self.bj.weight.data.zero_()
            self.wj.weight.data.uniform_(-1, 1)

        self.co_occurrence = co_occurrence.coalesce()
        # it is not very big
        self.coo_dense = self.co_occurrence.to_dense()
        self.x_max = x_max
        self.alpha = alpha
        # Placeholder. In future, we will make an abstract base class
        # which will have the below attribute so that all instances
        # carry their own loss.
        self.losses = []
        self._setup_indices()
        # This can only be set up later after the trainer has initialised
        self.device = None

    def _setup_indices(self):
        # Do some preprocessing to make looking up indices faster.
        # The co-occurrence matrix is a large array of pairs of indices
        # In the course of training, we will be given a list of
        # indices, and we need to find the pairs that are present.
        self.allindices = self.co_occurrence.indices()
        N = self.allindices.max() + 1
        # Store a dense array of which pairs are active
        # It is booleans so should be small even if there are a lot of tokens
        self.allpairs = torch.zeros((N, N), dtype=bool)
        self.allpairs[(self.allindices[0], self.allindices[1])] = True
        self.N = N

    def _set_device(self, device):
        self.device = device
        self.wi = self.wi.to(device)
        self.bi = self.bi.to(device)
        if self.double:
            self.wj = self.wj.to(device)
            self.bj = self.bj.to(device)
        self.co_occurrence = self.co_occurrence.to(device)
        self.coo_dense = self.coo_dense.to(device)
        self.allpairs = self.allpairs.to(device)

    @property
    def weights(self):
        if self.double:
            return self.wi.weight + self.wj.weight
        return self.wi.weight

    # implemented as such to be consistent with nn.Embeddings interface
    def forward(self, indices):
        return self.weights[indices]

    def _update(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        b_i = self.bi(i_indices).squeeze()
        if self.double:
            w_j = self.wj(j_indices)
            b_j = self.bj(j_indices).squeeze()
            x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        else:
            x = torch.sum(w_i, dim=1) + b_i
        return x

    def _loss_weights(self, x):
        # x: co_occurrence values
        wx = (x/self.x_max)**self.alpha
        wx = torch.min(wx, torch.ones_like(wx))
        return wx

    def loss(self, indices):
        # inputs are indexes, targets are actual embeddings
        # In the actual algorithm, "inputs" is the weight_func run on the
        # co-occurrence statistics.
        # loss = wmse_loss(weights_x, outputs, torch.log(x_ij))
        # not sure what it should be replaced by here
        # "targets" are the log of the co-occurrence statistics.
        # need to make every pair of indices that exist in co-occurrence file
        # Not every index will be represented in the co_occurrence matrix
        # To calculate glove loss, we will take all the pairs in the co-occurrence
        # that contain anything in the current set of indices.
        # There is a disconnect between the indices that are passed in here,
        # and the indices of all pairs in the co-occurrence matrix
        # containing those indices.
        indices = indices
        indices = indices.sort()[0]
        subset = self.allpairs[indices]
        if not torch.any(subset):
            self.losses = [0]
            return self.losses
        # now look up the indices of the existing pairs
        # it is faster to do the indexing into an array of bools
        # instead of the dense array
        subset_indices = torch.nonzero(subset)
        i_indices = indices[subset_indices[:, 0]]
        j_indices = subset_indices[:, 1]

        i_indices = i_indices
        j_indices = j_indices

        targets = self.coo_dense[(i_indices, j_indices)]
        weights_x = self._loss_weights(targets)
        current_weights = self._update(i_indices, j_indices)
        loss = weights_x * F.mse_loss(
            current_weights, torch.log(targets), reduction='none')
        # This is a feasible strategy for mapping indices -> pairs
        # Second strategy: Loop over all possible pairs
        # More computationally intensive
        # Allow this to be configurable?
        # Degrees of separation but only allow 1 or -1
        # -1 is use all indices
        # 1 is use indices
        # We may want to save this loss as an attribute on the layer object
        # Does Lightning have a way of representing layer specific losses
        # Define an interface by which we can return loss objects
        # probably stick with self.losses = [loss]
        # - a list - because then we can do +=
        loss = torch.mean(loss)
        self.losses = [loss]
        return self.losses

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # eventually, load this from a .pt file that contains all the
        # wi/wj/etc.
        raise NotImplementedError('Not yet implemented')


class GloveSimple(pl.LightningModule):

    def __init__(self, train_embeddings_file, batch_size, train_cooccurrence_file,
                 limit=None):
        # train_embeddings_file: the filename contaning the pre-trained weights
        # train_cooccurrence_file: the filename containing the co-occurrence statistics
        # that we want to match these embeddings to.
        super(GloveSimple, self).__init__()
        self.limit = limit
        self.train_data = torch.load(train_embeddings_file)
        self.train_cooccurrence = torch.load(train_cooccurrence_file)
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
        # Need to call _set_device later.
        # We cannot create the GloveLayer later, because then some initialisation
        # doesn't happen and the optimiser will blow up.
        self.glove_layer = GloveLayer(
            self.num_embeddings, self.embedding_dim,
            self.train_cooccurrence)

    def forward(self, indices):
        return self.glove_layer(indices)

    def training_step(self, batch, batch_idx):
        # if this isn't done explicitly it somehow never gets set automatically
        # by lightning
        if self.glove_layer.device is None:
            self.glove_layer._set_device(self.device)
        # input: indices of embedding
        # targets: target embeddings
        indices, targets = batch
        # Ideal outcome:
        # Pytorch object that is substitutable with
        # nn.Embedding
        # Subclass of nn.Embedding
        # but internally, we should give the option to use
        # a co-occurrence matrix
        out = self.glove_layer.weights[indices]
        loss = F.mse_loss(out, targets)
        glove_layer_loss = self.glove_layer.loss(indices)
        loss += glove_layer_loss[0]
        # How would this be used:
        # Take 2 domains with co-occurrence
        # Figure out intersection
        # Put these 2 layers into B-network
        # 2 branches which will connect at the top
        # Say we have L and R where L = openimages, R = something else
        # Top layer will combine audioset and openimage
        # and will be the "alignment layer"
        # Will take collective set of all concepts in both domains
        # The aligner layer will backpropagate down
        # to the respective embedding layers
        print(f"loss: {loss}")
        return loss

    def configure_optimizers(self):
        # Is there an equivalent configure_losses?
        opt = optim.Adam(self.parameters(), lr=0.005)
        return opt

    def train_dataloader(self):
        dataset = GloveEmbeddingsDataset(self.train_data, self.limit)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class GloveEmbeddingsDataset(Dataset):
    # Dataset for existing embedings

    def __init__(self, data, limit=None):
        # train_data contains wi.weight / wj.weight / bi.weight / bj.weight
        # for stability, the target is the wi + wj
        self.weights = data['wi.weight'] + data['wj.weight']
        if limit:
            self.weights = self.weights[:limit]
        nemb, dim = self.weights.shape
        # The dataset does not appear to need to be moved to GPU.
        # Lightning takes care of that
        self.x = torch.arange(nemb)
        self.y = self.weights
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
    # change to gpus=1 to use GPU. Otherwise CPU will be used
    trainer = pl.Trainer(gpus=0, max_epochs=100, progress_bar_refresh_rate=20)
    # Trainer must be created before model, because we need to detect
    # what we requested for GPU.

    model = GloveSimple('glove_audio.pt', batch_size=100,
                        train_cooccurrence_file='../audioset/co_occurrence_audio_all.pt')
    trainer.fit(model)


