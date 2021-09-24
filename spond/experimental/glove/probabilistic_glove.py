# probabilistic embeddings
import itertools
import pandas as pd
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F


import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import socket
if socket.gethostname().endswith('pals.ucl.ac.uk'):
    # set up pythonpath
    ppath = '/home/petra/spond'
    # set up data pth
    datapath = '/home/petra/data'
    gpu = True
else:
    ppath = '/opt/github.com/spond/spond/experimental'
    datapath = ppath
    gpu = False

sys.path.append(ppath)

from spond.experimental.glove.glove_layer import GloveEmbeddingsDataset


class ProbabilisticGloveLayer(nn.Embedding):

    # TODO: Is there a way to express constraints on weights other than
    # the nn.Functional.gradient_clipping ?
    # ANSWER: Yes, if you use Pyro with constraints.
    def __init__(self, num_embeddings, embedding_dim, co_occurrence,
                 # glove learning options
                 x_max=100, alpha=0.75,
                 seed=None, # if None means don't set
                 # whether or not to use the wi and wj
                 # if set to False, will use only wi
                 double=False,
                 # nn.Embedding options go here
                 padding_idx=None,
                 scale_grad_by_freq=None,
                 max_norm=None, norm_type=2,
                 sparse=False   # not supported- just here to keep interface
                 ):
        self.seed = seed
        if seed is not None:
            # internal import to allow setting seed before any other imports
            import pyro
            pyro.set_rng_seed(seed)
        # Internal import because we need to set seed first
        from pyro.distributions import MultivariateNormal
        # This is spurious; we won't actually be using any of the superclass
        # attributes, but we have to do this to get other things like the
        # registration of parameters to work.
        super(ProbabilisticGloveLayer, self).__init__(num_embeddings, embedding_dim,
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
        # double is not supported, but we keep the same API.
        assert not double, "Probabilistic embedding can only be used in single mode"
        # This assumes each dimension is independent of the others,
        # and that all the embeddings are independent of each other.
        # We express it as MV normal because this allows us to use a
        # non diagonal covariance matrix
        # try setting the variance low here instead
        # The output of these needs to be moved to GPU before use,
        # because there is currently no nice way to move a distribution to GPU.
        self.wi_dist = MultivariateNormal(
            torch.zeros((num_embeddings, embedding_dim)),
            # changing this to 1e-9 makes the embeddings converge to something
            # else than the pretrained
            torch.eye(embedding_dim)# * 1e-9
        )
        self.bi_dist = MultivariateNormal(
            torch.zeros((num_embeddings, 1)),
            torch.eye(1)
        )
        # Deterministic means for the weights and bias, that will be learnt
        # means will be used to transform the samples from the above wi/bi
        # samples.
        # Express them as embeddings because that sets up all the gradients
        # for backprop, and allows for easy indexing.
        self.wi_mu = nn.Embedding(num_embeddings, embedding_dim)
        self.wi_mu.weight.data.uniform_(-1, 1)
        self.bi_mu = nn.Embedding(num_embeddings, 1)
        self.bi_mu.weight.data.zero_()
        # wi_sigma = log(1 + exp(wi_rho)) to enforce positivity.
        self.wi_rho = nn.Embedding(num_embeddings, embedding_dim)
        # initialise the rho so softplus results in small values
        # 1e-9 - this appears to be about -20 so we have to re-center around -19?
        # except it doesn't work- re-centering just makes the means nowhere near
        self.wi_rho.weight.data.uniform_(-1, 1)
        self.bi_rho = nn.Embedding(num_embeddings, 1)
        self.bi_rho.weight.data.zero_()
        # using torch functions should ensure backprop is set up right
        self.softplus = nn.Softplus()
        #self.wi_sigma = softplus(self.wi_rho.weight) #torch.log(1 + torch.exp(self.wi_rho))
        #self.bi_sigma = softplus(self.bi_rho.weight) #torch.log(1 + torch.exp(self.bi_rho))

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

    @property
    def weight(self):
        return self.weights()

    def weights(self, n=1, squeeze=True):
        # we are taking one sample from each embedding distribution
        sample_shape = torch.Size([n])
        wi_eps = self.wi_dist.sample(sample_shape).type_as(self.wi_mu.weight.data)
        # TODO: Only because we have assumed a diagonal covariance matrix,
        # is the below elementwise multiplication (* rather than @).
        # If it was not diagonal, we would have to do matrix multiplication
        #wi = self.wi_mu + wi_eps * self.wi_sigma
        wi = (
                self.wi_mu.weight +
                # multiplying by 1e-9 below should have the same effect
                # as changing the wi_eps variance to 1e-9, but it doesn't.
                # multiplying here results in wi_mu converging very closely
                # to the deterministic embeddings, but the wi_sigma variance remains
                # the same as in the other case.
                wi_eps * self.softplus(self.wi_rho.weight) #* 1e-9
        )
        if squeeze:
            return wi.squeeze()
        else:
            return wi

    # implemented as such to be consistent with nn.Embeddings interface
    def forward(self, indices):
        return self.weights()(indices)

    def _update(self, i_indices, j_indices):
        # we need to do all the sampling here.
        # TODO: Not sure what to do with j_indices. Do we update the j_indices
        # TODO: Only because we have assumed a diagonal covariance matrix,
        # is the below elementwise multiplication (* rather than @).
        # If it was not diagonal, we would have to do matrix multiplication
        w_i = (
                self.wi_mu(i_indices) +
                self.wi_eps[i_indices] * self.softplus(self.wi_rho(i_indices))
        )

        b_i = (
                self.bi_mu(i_indices) +

                self.bi_eps[i_indices] * self.softplus(self.bi_rho(i_indices))
        ).squeeze()
        # If the double updating is not done, it takes a long time to converge.
        w_j = (
                self.wi_mu(j_indices) +
                self.wi_eps[j_indices] * self.softplus(self.wi_rho(j_indices))
        )
        b_j = (
                self.bi_mu(j_indices) +
                self.bi_eps[j_indices] * self.softplus(self.bi_rho(j_indices))
        ).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j
        return x

    def _init_samples(self):
        # On every 0th batch in an epoch, sample everything.
        sample_shape = torch.Size([])
        self.wi_eps = self.wi_dist.sample(sample_shape) #* 1e-9
        self.bi_eps = self.bi_dist.sample(sample_shape) #* 1e-9
        # This has to be done because there is currently no nice way to move
        # a Pyro distribution to GPU.
        # So we move whatever we sampled from it.
        template = self.wi_mu.weight.data
        self.wi_eps = self.wi_eps.type_as(template)
        self.bi_eps = self.bi_eps.type_as(template)

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
        indices = indices.sort()[0]
        subset = self.allpairs[indices]
        if not torch.any(subset):
            self.losses = [0]
            return self.losses
        # now look up the indices of the existing pairs
        # it is faster to do the indexing into an array of bools
        # instead of the dense array
        subset_indices = torch.nonzero(subset).type_as(indices)
        i_indices = indices[subset_indices[:, 0]]
        j_indices = subset_indices[:, 1]

        targets = self.coo_dense[(i_indices, j_indices)]
        weights_x = self._loss_weights(targets)
        current_weights = self._update(i_indices, j_indices)
        # put everything on the right device
        weights_x = weights_x.type_as(current_weights)
        targets = targets.type_as(current_weights)

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

    def entropy(self):
        # Calculate entropy based on the learnt rho (which must be transformed
        # to a variance using softplus)
        # entropy of a MV Gaussian =
        # 0.5 * N * ln (2 * pi * e) + 0.5 * ln (det C)
        N = self.embedding_dim
        C = self.softplus(self.wi_rho.weight)
        # diagonal covariance so just multiply all the items to get determinant
        # convert to log space so we can add across the dimensions
        # We don't need to convert back to exp because we need log det C anyway
        logdetC = torch.log(C).sum(axis=1)
        entropy = 0.5*(N*np.log(2*np.pi * np.e) + logdetC)
        return entropy

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # eventually, load this from a .pt file that contains all the
        # wi/wj/etc.
        raise NotImplementedError('Not yet implemented')


class ProbabilisticGlove(pl.LightningModule):

    # This class is meant to provide stochastic Glove embeddings.
    # It is given a deterministically-generated embeddings file,
    # as well as the file of cooccurrence data.
    def __init__(self, train_embeddings_file, batch_size, train_cooccurrence_file,
                 use_pretrained=False,
                 seed=None):
        # train_embeddings_file: the filename contaning the pre-trained weights
        #                        from a determistic Glove run.
        # train_cooccurrence_file: the filename containing the co-occurrence statistics
        # that we want to match these embeddings to.
        # use_pretrained: If True, pretrained embeddings will be used as a reference
        super(ProbabilisticGlove, self).__init__()
        self.seed = seed
        self.use_pretrained = use_pretrained
        self.train_embeddings_file = train_embeddings_file
        self.train_cooccurrence_file = train_cooccurrence_file
        self.train_data = torch.load(train_embeddings_file, map_location=torch.device('cpu'))
        self.train_cooccurrence = torch.load(train_cooccurrence_file)
        self.batch_size = batch_size
        nemb, dim = self.train_data['wi.weight'].shape
        self.pretrained_embeddings = (
                self.train_data['wi.weight'] +
                self.train_data['wj.weight']
        )
        self.num_embeddings = nemb
        self.embedding_dim = dim
        self.glove_layer = ProbabilisticGloveLayer(
            self.num_embeddings, self.embedding_dim,
            self.train_cooccurrence, seed=self.seed)

    # def additional_state(self):
    #     # return dictionary of things that were passed to constructor
    #     # should contain everything necessary to replicate a model.
    #     # we don't save things like the actual training data and so on
    #     # obviously this means that when the model is loaded,
    #     # the appropriate training file must be present.
    #     state = dict(
    #         seed=self.seed,
    #         train_embeddings_file=self.train_embeddings_file,
    #         train_cooccurrence_file=self.train_cooccurrence_file,
    #         use_pretrained=self.use_pretrained,
    #         batch_size=self.batch_size,
    #     )
    #     return state

    # def save(self, filename):
    #     # this is the torch model stuff
    #     state = self.state_dict()
    #     # this is the custom state like embeddings file name, etc
    #     state.update(self.additional_state())
    #     torch.save(state, filename)

    # @classmethod
    # def load(cls, filename, device='cpu'):
    #     state = torch.load(filename, map_location=device)
    #     # get the items that would have been passed to the constructor
    #     additional_state = {}
    #     items = (
    #         'seed', 'train_embeddings_file',
    #         'train_cooccurrence_file', 'batch_size', 'use_pretrained',
    #     )
    #     for item in items:
    #         additional_state[item] = state.pop(item)
    #     instance = cls(**additional_state)
    #     instance.load_state_dict(state)
    #     return instance

    def forward(self, indices):
        return self.glove_layer.weights()(indices)

    def training_step(self, batch, batch_idx):
        # If the batch_idx is 0, then we want to sample everything at once
        # if we do multiple samples, end up with torch complaining we are
        # trying to do backprop more than once.
        if batch_idx == 0:
            self.glove_layer._init_samples()

        # input: indices of embedding
        # targets: target embeddings
        indices, targets = batch
        # Ideal outcome:
        # Pytorch object that is substitutable with
        # nn.Embedding
        # Subclass of nn.Embedding
        # but internally, we should give the option to use
        # a co-occurrence matrix
        loss = 0
        if targets.shape[1] > 0:
            out = self.glove_layer.weights()[indices]
            loss = F.mse_loss(out, targets)
        glove_layer_loss = self.glove_layer.loss(indices)
        loss += glove_layer_loss[0]
        # How would this be used:
        # Take 2 domains with co-occurrence
        # Figure out intersection
        # Put these 2 layers into B-network
        # 2 branches which will connect at the top
        # Say we have L and R where L = openimages, R = something else
        # Top layer will combine audioset and openimages
        # and will be the "alignment layer"
        # Will take collective set of all concepts in both domains
        # The aligner layer will backpropagate down
        # to the respective embedding layers
        print(f"loss: {loss}")
        return loss

    def configure_optimizers(self):
        # Is there an equivalent configure_losses?
        # TODO: This is a torch optimiser. If we change to Pyro,
        # have to change to Pyro optimiser.
        # TODO: The learning rate probably needs to change,
        # because we are now sampling and it's going to be very stochastic.
        opt = optim.Adam(self.parameters(), lr=0.01)
        return opt

    def train_dataloader(self):
        train_data = self.train_data if self.use_pretrained else None
        dataset = GloveEmbeddingsDataset(train_data, self.num_embeddings)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class Similarity:
    # this class will take a root directory and a class and a list of ints (the seeds)
    # and it will load the models contained in that directory
    # and calculate the similarity matrices required.
    # The convention is that the models will be stored in a directory/file structure
    # as follows:
    # dirname/classname/classname_seed.pt
    # for example
    # dirname/ProbabilisticGlove/ProbabilisticGlove_2.pt
    # is the ProbabilisticGlove model run with seed set to 2.
    def __init__(self, dirname, clsobj, seedvalues, tag=None):
        self.dirname = dirname
        self.clsobj = clsobj
        self.seedvalues = seedvalues
        self.models = {}
        self.tag = '' if tag is None else f"{tag}_"

    def _load(self, seed):
        # load the model with the corresponding seed.
        # don't load all at once as this can lead to out of memory
        clsname = self.clsobj.__name__
        filename = os.path.join(dirname, clsname, f'{self.tag}{clsname}_{seed}.pt')
        model = torch.load(filename) #self.clsobj.load(filename)
        return model

    def sim_means(self, kernel, outfile, mode='a', mask=None):
        # kernel: callable that takes 2 arrays and returns similarity matrix
        # outfile: target file name
        # mask: if passed, should be a sequence of integers for which the
        # similarity will be calculated. It is up to the user to keep track
        # of what the final output indices mean.
        # Similarity will be calculated for each seed using the specified metric
        # and stored in `outfile`
        store = pd.HDFStore(outfile, mode=mode)
        for seed in self.seedvalues:
            model = self._load(seed)
            values = model.glove_layer.wi_mu.weight.detach().numpy()
            if mask is not None:
                values = values[mask]
            thisvalue = kernel(values, values)
            store[str(seed)] = pd.DataFrame(thisvalue)
        store.close()


if __name__ == '__main__':
    import os
    import kernels
    import gc
    import sys
    from spond.experimental.openimages.readfile import readlabels

    # You have to run this manually once for openimages and once for audioset.
    tag = 'audioset'  # 'openimages' #

    if tag == 'openimages':
        input_embeddings = 'glove_imgs.pt'
        co_occurrence = 'co_occurrence.pt'
        max_epochs = 250
    else:
        input_embeddings = 'glove_audio.pt'
        co_occurrence = 'co_occurrence_audio_all.pt'
        max_epochs = 2000

    seeds = (1, 2, )#3, 4, 5, 6, 7, 8, 9, 10)
    for seed in seeds:
        # change to gpus=1 to use GPU. Otherwise CPU will be used
        # needs to be higher for audioset.
        trainer = pl.Trainer(gpus=int(gpu), max_epochs=max_epochs, progress_bar_refresh_rate=20)
        # Trainer must be created before model, because we need to detect
        # what we requested for GPU.
        model = ProbabilisticGlove(os.path.join(datapath, tag, input_embeddings),
                                   use_pretrained=False,
                                   batch_size=500,
                                   seed=seed,
                                   train_cooccurrence_file=os.path.join(datapath, tag, co_occurrence)
                                   )

        trainer.fit(model)
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', tag)
        clsname = model.__class__.__name__
        outdir = os.path.join(outdir, clsname)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        outfile = os.path.join(outdir, f'{tag}_{clsname}_{seed}.pt')
        torch.save(model, outfile)
        del model
        del trainer
        print(f"finished seed {seed}")
        torch.cuda.empty_cache()
        gc.collect()
    if tag == 'openimages':
        labelsfn = os.path.join(datapath, tag, 'oidv6-class-descriptions.csv')
    else:
        labelsfn = os.path.join(datapath, tag, 'class_labels.csv')
    labels, names = readlabels(labelsfn, rootdir=None)
    dirname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results', tag)
    sim = Similarity(dirname, ProbabilisticGlove, seedvalues=seeds,  tag=tag)
    # The dot product similarity of the learned means will be saved in the file {tag}_means_dot.hdf5
    # This will be used in analyse.py to generate various other things
    sim.sim_means(kernels.dot, os.path.join(dirname, 'ProbabilisticGlove', f'{tag}_means_dot.hdf5'), mask=None, mode='w')


