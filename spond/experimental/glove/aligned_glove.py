# This module will contain 2 ProbabilisticGlove layers that will be
# trained from their separate co-occurrence matrices.
# Module is not called aligner.py to avoid clashing with the existing module
# of that name. The two may be resolved later.

import pandas as pd
import sys

import itertools
import math
import numpy as np
import pandas as pd
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from spond.experimental.glove.probabilistic_glove import ProbabilisticGloveLayer
from spond.experimental.glove.glove_layer import GloveLayer
from spond.models.mlp import MLP
from spond.metrics.mapping_accuracy import mapping_accuracy
from spond.experimental.openimages.readfile import readlabels

# python setup.py install from https://github.com/josipd/torch-two-sample
from torch_two_sample import statistics_diff as sd

# python setup.py install from https://github.com/PythonOT/POT
import ot

# hidden layer size, somewhat arbitrarily choose 100
HIDDEN = 100


# Various currently unused loss functions that were tried
def energy_loss(mapped, target, device='cpu'):
    nx = mapped.shape[0]
    ny = target.shape[0]
    es = sd.EnergyStatistic(nx, ny)
    loss = es(mapped, target)
    return 100*loss


def fr_loss(mapped, target, device='cpu', alpha=0.01):
    # fr_loss_gyx = fr_loss(y_mapped[y_samples], self.x_emb.weight[x_samples], device=self.device,
    #                       alpha=0.01)
    # losses['fr_x'] = fr_loss_gyx
    #
    # fr_loss_fxy = fr_loss(x_mapped[x_samples], self.y_emb.weight[y_samples], device=self.device,
    #                       alpha=0.01)
    # losses['fr_y'] = fr_loss_fxy

    mapped = mapped
    target = target
    nx = mapped.shape[0]
    ny = target.shape[0]
    fr = sd.SmoothFRStatistic(nx, ny, cuda=True, compute_t_stat=False)
    loss = fr(mapped, target, [alpha])
    return loss.to(device)


def knn_loss(mapped, target, device='cpu', alpha=0.01, k=1):

    # samplesize = 5000
    # x_sz = min(samplesize, self.x_n)
    # y_sz = min(samplesize, self.y_n)
    # x_samples = torch.randint(low=0, high=self.x_n, size=(x_sz,))
    # y_samples = torch.randint(low=0, high=self.y_n, size=(y_sz,))
    #
    # knn_loss_gyx = knn_loss(y_mapped[y_samples], self.x_emb.weight[x_samples], device=self.device,
    # alpha=0.01, k=1)
    # losses['knn_x'] = knn_loss_gyx
    #
    # knn_loss_fxy = knn_loss(x_mapped[x_samples], self.y_emb.weight[y_samples], device=self.device,
    # alpha=0.01, k=1)
    # losses['knn_y'] = knn_loss_fxy

    nx = mapped.shape[0]
    ny = target.shape[0]
    mapped = mapped.to(device)
    target = target.to(device)
    cuda = device != 'cpu'
    knn = sd.SmoothKNNStatistic(nx, ny, cuda=cuda, k=k)
    loss = knn(mapped, target, alphas=[alpha])
    return loss.to(device)




def ot_loss(mapped, target, device='cpu'):
    reg = 5
    nx = mapped.shape[0]
    ny = target.shape[0]
    ab = torch.ones(nx)/nx
    ab = ab.to(device)
    M = ot.dist(mapped, target)   # euclidean by default
    loss = ot.sinkhorn2(ab, ab, M, reg)
    return  loss


def procwass_loss(mapped, target, R, device='cpu', reg=0.025):
    # # use Procrustes Wasserstein loss.
    # Ux, Sx, VTx = torch.linalg.svd(self.Rx)
    # self.Rx = torch.mm(Ux, VTx)
    # pw_loss_x = procwass_loss(y_mapped[y_intersect],
    #                           self.x_emb.weight[x_intersect],
    #                           self.Rx, device=self.device,
    #                           reg=20)
    # losses['pw_x'] = pw_loss_x
    # Uy, Sy, VTy = torch.linalg.svd(self.Ry)
    # self.Ry = torch.mm(Uy, VTy)
    # pw_loss_y = procwass_loss(x_mapped[x_intersect],
    #                           self.y_emb.weight[y_intersect],
    #                           self.Ry, device=self.device,
    #                           reg=20)
    # losses['pw_y'] = pw_loss_y
    #

    #  Procrustes / Wasserstein loss
    # R must be saved over epochs and initialised by calling convex_init
    # mapped, target must be same size
    C = -torch.mm(torch.mm(mapped, R), target.T)
    n = mapped.shape[0]
    onesn = torch.ones(n).to(device)
    P = ot.sinkhorn(onesn, onesn, C.detach(), reg, stopThr=1e-3)
    loss = (
            1000 * torch.linalg.norm(torch.mm(mapped, R) - torch.mm(P, target))/n
    )
    return loss


def mmd_loss(mapped, target, device='cpu', alpha=2):
    nx = mapped.shape[0]
    ny = target.shape[0]
    mmd = sd.MMDStatistic(nx, ny)
    alphas = [alpha]
    loss = mmd(mapped, target, alphas)
    return loss


class AlignedGloveLayer(nn.Module):

    def __init__(self,
                 x_cooc,                # co-occurrence matrix for x
                 x_embedding_dim,       # dimension of x
                 y_cooc,                # co-occurrence matrix for y
                 y_embedding_dim,       # dimension of y
                 index_map,              # list of pairs that map a concept in x
                 # to a concept in y
                 reg=0,      # L2 regularisation parameter
                 mmd=1.0,    # MMD loss scaling
                 seed=None,
                 probabilistic=False,     # If set, use ProbabilisticGloveLayer
                 supervised=True          # If set, will use the index map
                 # and supervised losses will be included,
                 ):
        super(AlignedGloveLayer, self).__init__()
        self.seed = seed
        self.reg = reg
        self.mmd = mmd
        self.probabilistic = probabilistic
        x_nconcepts = x_cooc.size()[0]
        y_nconcepts = y_cooc.size()[0]
        kws = {}
        if seed is not None:
            if probabilistic:
                kws['seed'] = seed
            torch.manual_seed(seed)
        self.supervised = supervised

        cls = ProbabilisticGloveLayer if probabilistic else GloveLayer
        self.x_emb = cls(x_nconcepts, x_embedding_dim, x_cooc, **kws)
        self.y_emb = cls(y_nconcepts, y_embedding_dim, y_cooc, **kws)
        self.x_n = x_nconcepts
        self.y_n = y_nconcepts
        self.index_map = index_map
        # This is stored just for speed
        self.index_map_dict = dict(index_map)
        self.rev_index_map_dict = {v: k for k, v in self.index_map_dict.items()}

        # build the MLPs that are the aligner layers
        # f(x) --> y
        self.fx = MLP(x_embedding_dim, HIDDEN, output_size=y_embedding_dim)
        # g(y) --> x
        self.gy = MLP(y_embedding_dim, HIDDEN, output_size=x_embedding_dim)
        self.losses = []
        # external flag set by other classes
        self.mse_cycle_loss = False

    def _init_samples(self):
        # TODO: not sure how this will work with different dataset sizes
        # for x and y.
        if self.probabilistic:
            self.x_emb._init_samples()
            self.y_emb._init_samples()

    def forward(self, indices):
        # indices are a tuple of x and y index
        x_ind, y_ind = indices
        losses = self.loss(x_ind, y_ind)
        return losses

    def loss(self, x_inds, y_inds, nsamples=10):
        # x_inds and y_inds are sequences of the x and y indices that form
        # this minibatch.
        # The loss contains the following items:
        # For all concepts:
        # 1. Glove loss for both x and y embeddings
        # recall that ProbabilisticGloveLayer will return a list of loss
        # There is no MSE loss here like there is in GloveLayer and
        # ProbabilisticGlove, because this is now no longer supervised.
        # We are not trying to train to match deterministic embeddings.
        losses = {}
        # scale glove_x by the ratio of numbers of concepts
        losses['glove_x'] = (self.x_n / self.y_n *
                             self.x_emb.loss(x_inds)[0])
        losses['glove_y'] = self.y_emb.loss(y_inds)[0]

        x = self.x_emb.weights(n=nsamples, squeeze=False)
        y = self.y_emb.weights(n=nsamples, squeeze=False)

        xdim = x.shape[-1]
        ydim = y.shape[-1]

        x_mapped = self.fx(x)
        y_mapped = self.gy(y)
        # calculate cycle loss: g(f(x)) - x
        x_rt = self.gy(x_mapped)
        fx_diff = x_rt - x
        fx_diff = fx_diff.reshape((-1, xdim))
        # This is the cycle loss: f(g(y)) - y
        y_rt = self.fx(y_mapped)
        gy_diff = y_rt - y
        gy_diff = gy_diff.reshape((-1, ydim))
        if self.mse_cycle_loss:
            cycle_gy_loss = torch.einsum('ij,ij->i', gy_diff, gy_diff).mean()
        else:
            cycle_gy_loss = torch.sqrt(torch.einsum('ij,ij->i', gy_diff, gy_diff)).mean() #sum()

        losses['cycle_x'] = cycle_gy_loss

        # other cycle loss: |f(g(x)) - x|
        if self.mse_cycle_loss:   # This was used by the MAGAN implementation
            cycle_fx_loss = torch.einsum('ij,ij->i', fx_diff, fx_diff).mean() #sum()
        else:
            cycle_fx_loss = torch.sqrt(torch.einsum('ij,ij->i', fx_diff, fx_diff)).mean() #sum()
        losses['cycle_y'] = cycle_fx_loss
        # For concepts that exist in both domains:
        # The intersection will always be trained for supervised loss.
        # This code relies heavily on the fact that the index_map is small,
        # therefore using numpy operations is fast.
        x_intersect = self.index_map[:, 0]
        y_intersect = self.index_map[:, 1]
        x_samples = x_intersect
        y_samples = y_intersect

        if self.probabilistic and self.mmd > 0:
            # # MMD on only the intersection will converge, but not to 100% accuracy.
            # alpha = 0.15 for x and 0.05 for y were obtained from calculating the median at each batch,
            # converting to alpha and running with it until convergence
            mmd_loss_gyx = mmd_loss(
                y_mapped[:, y_samples].reshape((-1, xdim)), x[:, x_samples].reshape((-1, xdim)),
                alpha=0.1)
            losses['mmd_x_intersect'] = self.mmd * mmd_loss_gyx

            mmd_loss_fxy = mmd_loss(
                x_mapped[:, x_samples].reshape((-1, ydim)), y[:, y_samples].reshape((-1, ydim)),
                alpha=0.1)
            losses['mmd_y_intersect'] = self.mmd * mmd_loss_fxy

        if self.supervised:
            sup_diff_y = x_mapped[:, x_intersect] - y[:, y_intersect]
            sup_diff_y = sup_diff_y.reshape((-1, ydim))
            sup_loss_y = torch.sqrt(torch.einsum('ij,ij->i', sup_diff_y, sup_diff_y)).mean()

            losses['supervised_y'] = sup_loss_y

            sup_diff_x = y_mapped[:, y_intersect] - x[:, x_intersect]
            sup_diff_x = sup_diff_x.reshape((-1, xdim))
            sup_loss_x = torch.sqrt(torch.einsum('ij,ij->i', sup_diff_x, sup_diff_x)).mean()
            losses['supervised_x'] = sup_loss_x
        # This section was used when trying unsupervised variants
        #else:
        #    fr_loss_x = fr_loss(
        #        y_mapped[:, y_samples].reshape((-1, xdim)),
        #        x[:, x_samples].reshape((-1, xdim)), device=self.device,
        #        alpha=0.01
        #    )
        #    fr_loss_y = fr_loss(
        #        x_mapped[:, x_samples].reshape((-1, ydim)),
        #        x[:, x_samples].reshape((-1, ydim)), device=self.device,
        #        alpha=0.01
        #    )
        #    losses['fr_loss_x'] = fr_loss_x
        #    losses['fr_loss_y'] = fr_loss_y
        if self.reg > 0:
            # L2 regulariation.
            # Had to be done like this, explicitly,
            # because otherwise 2 optimisers had to be defined
            # which was very fiddly
            mlp_params = list(self.fx.parameters()) + list(self.gy.parameters())
            l2_norm = sum([torch.norm(param, p=2) for param in mlp_params])
            losses['l2_reg'] = self.reg * l2_norm

        return losses


class AlignedGlove(pl.LightningModule):

    # I decided to use the labels files as inputs because then
    # all the code for taking indices can be done inside this class.
    # For this to work, the indices of the entries in the *_cooc_file must
    # match the order of the labels.
    # That is, if x_cooc_file corresponds to the openimages file,
    # the first row of the cooccurrence matrix from this file
    # must correspond to the first label/name in x_labels_file.
    # Strictly speaking, merged_labels_file is not necessary as we could
    # merge the files internally too. It will be used to generate the
    # indices of corresponding concepts (eg. index of "Bird" in openimages and
    # index of "Bird" in audioset).
    def __init__(self,
                 batch_size,
                 data,  # DataDict class
                 x_embedding_dim,  # dimension of x
                 y_embedding_dim,  # dimension of y
                 reg=0, # regularisation parameter for MLPs in aligner
                 mmd=1.0, # factor with which to scale MMD loss
                 probabilistic=False,  # whether to use probabilistic layers
                 seed=None,
                 supervised=True,
                 max_epochs=None,
                 save_flag=True,
                 outdir='',  # directory where to save
                 ):
        super(AlignedGlove, self).__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.data = data
        self.x_embedding_dim = x_embedding_dim
        self.y_embedding_dim = y_embedding_dim
        self.probabilistic = probabilistic
        self.supervised = supervised
        self.max_epochs = max_epochs
        self.save_flag = save_flag
        self.outdir = outdir
        self.aligner = AlignedGloveLayer(
            self.data.x_cooc,
            self.x_embedding_dim,
            self.data.y_cooc,
            self.y_embedding_dim,
            self.data.index_map,
            seed=seed,
            probabilistic=probabilistic,
            supervised=supervised,
            reg=reg,
            mmd=mmd
        )
        self.losses = []
        self.epochs = 0
        # mean of x and y accuracies
        self.last_accs  = [] # store by epoch
        self.last_acc_min = 0.0

        # for saving. The tag is generated from the hyperparameters.
        tag = 'probabilistic' if probabilistic else 'deterministic'
        tag = f'{tag}_sup' if supervised else f'{tag}_unsup'
        tag = f'{tag}_{reg}' if reg > 0 else tag
        tag = f'{tag}_mmd{mmd}' if not np.isclose(mmd, 1) else tag
        self.filename = f'{tag}_{max_epochs}_AlignedGlove_{seed}.pt'
        self.min_epoch = 0

    def on_train_end(self):
        # PL hook
        df = pd.DataFrame(self.losses)
        index = pd.Index(np.linspace(0, self.epochs, num=len(df.index)+1))
        df.index = index[:-1]
        self.losses = df
        if self.save_flag:
            torch.save(self, os.path.join(self.outdir, self.filename.replace(".pt", "_last.pt")))

    def on_train_epoch_end(self, *args, **kwargs):
        self.epochs += 1

        epoch_mean_acc = np.mean(self.last_accs)
        # only save if mean_accuracy is more than that on last epoch
        if epoch_mean_acc >= self.last_acc_min:
            # don't bother saving if accuracy is below 90%
            if epoch_mean_acc >= 0.90 and self.save_flag:
                print(f'Would save to {self.filename}: last_acc_min={self.last_acc_min}, epoch_mean_acc={epoch_mean_acc}')
                torch.save(self, os.path.join(self.outdir, self.filename))
            self.last_acc_min = epoch_mean_acc
            self.min_epoch = self.epochs
        # reset the array
        self.last_accs = []

    def forward(self, indices):
        # indices are a tuple of x and y index
        x_ind, y_ind = indices
        losses = self.aligner.loss(x_ind, y_ind)
        return losses

    def training_step(self, batch, batch_idx):
        # init samples every batch and not just on batch_idx = 0
        self.aligner._init_samples()

        # indices: the indices of the items present in this batch
        #          essentially meaningless because x_ind and y_ind are
        #          more important
        # x_ind: indices of x embeddings to be used in this batch
        # y_ind: indices of y embeddings to be used in this batch
        indices, xy_indices = batch
        x_ind = xy_indices[:, 0]
        y_ind = xy_indices[:, 1]
        # forward step
        losses = self.forward((x_ind, y_ind))

        x_acc, y_acc = self.evaluate(self.device)

        accdict = dict(acc_x=x_acc, acc_y=y_acc)

        mean_acc = np.mean([x_acc, y_acc])

        self.last_accs = np.concatenate([self.last_accs, [mean_acc]])

        print(f"losses: {losses}")
        print(f"accuracy: {accdict}")

        loss = sum(losses.values())

        # detach or the graph will become too big
        lossdict = {
            k: v.detach().cpu().numpy().item()
            for k, v in losses.items()
        }
        lossdict.update(accdict)
        self.losses.append(pd.Series(lossdict))
        return loss

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=0.01)
        return opt

    def train_dataloader(self):
        dataset = GloveDualDataset(self.data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def evaluate(self, device='cpu'):
        # test how good the alignment is by computing accuracy both ways.
        al = self.aligner
        x_intersect = al.index_map[:, 0]
        y_intersect = al.index_map[:, 1]
        n = len(x_intersect)
        x_emb = al.x_emb.weights(n=1, squeeze=True).to(device)
        y_emb = al.y_emb.weights(n=1, squeeze=True).to(device)
        fx = al.fx.to(device)(x_emb)
        fx_dist = torch.cdist(fx[x_intersect], y_emb[y_intersect],
                              compute_mode="donot_use_mm_for_euclid_dist").to(device)

        _, nn_inds_x = fx_dist.topk(1, dim=1, largest=False)
        fx_acc = nn_inds_x.squeeze() == torch.arange(n).to(device)

        # and we want gy[y_intersect] to be close to x_emb.weight[x_intersect]
        gy = al.gy.to(device)(y_emb)
        gy_dist = torch.cdist(gy[y_intersect], x_emb[x_intersect],
                              compute_mode="donot_use_mm_for_euclid_dist").to(device)
        _, nn_inds_y = gy_dist.topk(1, dim=1, largest=False)
        #gy_acc = nn_inds_y.squeeze() == torch.tensor(x_intersect).to(device)
        gy_acc = nn_inds_y.squeeze() == torch.arange(n).to(device)
        # (x accuracy, y accuracy)
        return [gy_acc.float().mean().item(), fx_acc.float().mean().item()]


class GloveDualDataset(Dataset):

    def __init__(self, data):
        self.data = data
        # Internally, Lightning will make index samples based on whatever
        # is returned by self.__len__().
        # This means that if len(self) = 2000 and batch size is 300,
        # Lightning will do whatever is necessary to make N batches of
        # 300 random indexes from 0 to 1999.
        #
        # We need to make sure all concepts are trained regardless of whether
        # they appear in audio, images, or both.
        #
        # There are many more openimages items than audioset
        # If we just sample from the total set of indices, we'll end up
        # with very imbalanced training.
        # There are 526 audio concepts and 19000+ openimage ones.
        #
        # Suppose we have batch size 100
        # In one epoch we have to go through ~1900 openimage batches and
        # 6 audio batches
        #
        # I think the right thing to do is to "unroll" the audio dataset
        # to make the total the same length as the openimage dataset.
        # Then, randomise the unrolled audio indexes so that
        # the pairing of openimage index with audio index is nondeterministic
        #
        # Then, one batch will be 100 pairs of indices from both sets,
        # and we should go randomly through multiple iterations of the
        # smaller dataset.
        x_n = self.data.x_n
        y_n = self.data.y_n
        self.x_n = x_n
        self.y_n = y_n
        self.N = max(x_n, y_n)

        larger = x_n if x_n > y_n else y_n
        smaller = y_n if x_n == larger else x_n
        self.N = larger
        times = math.ceil(larger/smaller)

        smaller_inds = np.repeat([np.arange(smaller)], times, axis=0).ravel()[:larger]
        np.random.shuffle(smaller_inds)
        self.inds = torch.arange(self.N)
        # TODO: fix this, there's an assumption that x is larger than y
        self.out = torch.vstack([torch.arange(larger), torch.tensor(smaller_inds)]).T

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # This must return whatever we map idx to internally.
        # It doesn't matter what is returned as long as we
        # unpack it correctly inside the training loop.
        inds = self.inds[idx]
        out = self.out[idx]
        return inds, out


class DataDictionary:
    # This class takes the co-occurrence matrices and labels files
    # and does some sanity checks, as well as storing various useful things.

    def __init__(self,
                 x_cooc,      # co-occurrence matrix for x
                 x_labels_file,    # full filepath or handle containing all x labels to names
                 y_cooc,      # co-occurrence matrix for y
                 y_labels_file,    # full filepath or handle containing all y labels to names
                 all_labels_file,        # full filepath or handle containing all labels to names
                 intersection_plus=None  # controls whether to load only the intersection plus a fixed number
                 # of concepts. None means do everything.
                 # Must be an int otherwise; this many samples from the
                 # remainder of the concepts will be included
                 ):

        x_nconcepts = x_cooc.size()[0]
        y_nconcepts = y_cooc.size()[0]

        self.intersection_plus = intersection_plus

        self.x_labels_file = x_labels_file
        self.y_labels_file = y_labels_file
        self.x_cooc_orig = x_cooc
        self.y_cooc_orig = y_cooc

        self.x_n = x_nconcepts
        self.y_n = y_nconcepts
        self.all_labels_file = all_labels_file
        # all_labels = label to index in co-occurrence
        # all_names = label to concept name
        all_labels, all_names = readlabels(all_labels_file, rootdir=None)
        # some of these mappings may be useful to expose
        name_to_label = {v: k for k, v in all_names.items()}
        index_to_label = {v: k for k, v in all_labels.items()}
        index_to_name = {v: all_names[k] for k, v in all_labels.items()}
        name_to_index = {v: k for k, v in index_to_name.items()}

        self.index_to_name = index_to_name

        # x_labels is dictionary of label to index
        # x_names is dictionary of label to name
        x_labels, x_names = readlabels(x_labels_file, rootdir=None)
        assert len(x_labels) == x_nconcepts, (
            f"x co-occurrence does not contain the same number of concepts as the labels file: "
            f"\nExpected {x_nconcepts} but got {len(x_labels)}"
        )

        self.x_labels = x_labels
        self.x_names = x_names

        # same applies to y_labels and y_names
        y_labels, y_names = readlabels(y_labels_file, rootdir=None)
        assert len(y_labels) == y_nconcepts, (
            f"y co-occurrence does not contain the same number of concepts as the labels file: "
            f"\nExpected {y_nconcepts} but got {len(y_labels)}"
        )

        self.y_labels = y_labels
        self.y_names = y_names

        y_name_to_label = {v: k for k, v in y_names.items()}
        intersection = {}

        for x_label, x_name in x_names.items():
            if x_label in y_names:
                # we have to use labels as the intersection,
                # because there are multiple labels with the same name.
                # for example /m/07qcpgn is Tap in audioset, meaning the sound Tap
                # but /m/02jz0l is Tap in openimages meaning the object Tap.
                intersection[x_label] = (x_labels[x_label], y_labels[y_name_to_label[x_name]])
        # Tuple of (index in all, index in x, index in y)
        # TODO: ugh, too many levels of indirection, clean up later.
        index_map = list(intersection.values())
        x_indexes = torch.tensor([all_labels[label] for label in x_labels])
        y_indexes = torch.tensor([all_labels[label] for label in y_labels])

        self.intersection_names = intersection

        # universe: keys = labels, values = index into universe
        self.all_labels = all_labels
        # universe: keys = labels, values = names
        self.all_names = all_names

        # indexes of x concepts in the universe
        self.x_indexes = x_indexes
        # indexes of y concepts in the universe
        self.y_indexes = y_indexes
        # given an index in the universe, stores the mapping from
        # index into universe to index in x and y files <--- note, it's in the files not the co-occurrence
        # eg.  if universe index 5 is Cat and x index 0 is Cat and
        # y index 77 is Cat then
        # self.union_indexes contains
        #  (5, 0, 77)
        self.intersection_indexes = torch.tensor([
            (
                all_labels[label],
                x_labels[label],
                y_labels[label]
            )
            for label in intersection
        ])

        index_map = self.intersection_indexes[:, 1:].numpy()
        # if intersection_plus was set, work out what to actually include in the co-occurrence.
        # Note we do this only after checking the labels files for consistency.
        # We have to change the intersection indexes / index map to take this into account.
        if intersection_plus is not None:
            assert isinstance(intersection_plus, int), "Pass an int for intersection_plus"
            x_intersect = index_map[:, 0]
            y_intersect = index_map[:, 1]
            # if it's 0, then use only the intersection of concepts.
            x_included = x_intersect
            y_included = y_intersect
            if intersection_plus > 0:
                # otherwise, we have to randomly sample some further items
                x_available = set(x_indexes).difference(set(x_intersect))
                y_available = set(y_indexes).difference(set(y_intersect))
                x_additional = np.random.choice(list(x_available), intersection_plus, replace=False)
                y_additional = np.random.choice(list(y_available), intersection_plus, replace=False)
                x_included = np.hstack([x_additional, x_included])
                y_included = np.hstack([y_additional, y_included])
            self.x_cooc = x_cooc.to_dense()[x_included].T[x_included].T.to_sparse()
            self.y_cooc = y_cooc.to_dense()[y_included].T[y_included].T.to_sparse()
            # Now, we have to recreate the index map based on the new indexes,
            # and keep a mapping.
            self.index_map = np.vstack([np.arange(len(x_intersect)), np.arange(len(y_intersect))]).T
            self.x_to_orig = {k: v for k, v in enumerate(x_included)}
            self.y_to_orig = {k: v for k, v in enumerate(y_included)}
            self.x_n = len(x_included)
            self.y_n = len(y_included)
        else:
            self.x_cooc = self.x_cooc_orig
            self.y_cooc = self.y_cooc_orig
            self.index_map = index_map
            self.x_to_orig = dict(zip(self.x_indexes, self.x_indexes))
            self.y_to_orig = dict(zip(self.y_indexes, self.y_indexes))

    def state_dict(self):
        # this will only work if all the files are strings
        check_files = (self.x_labels_file, self.y_labels_file, self.all_labels_file)
        if not all([isinstance(f, str) for f in check_files]):
            raise AssertionError("Can only persist this item if all filenames are strings")

        return dict(
            x_cooc=self.x_cooc,      # co-occurrence matrix for x
            x_labels_file=self.x_labels_file,    # full filepath or handle containing all x labels to names
            y_cooc=self.y_cooc,      # co-occurrence matrix for y
            y_labels_file=self.y_labels_file,    # full filepath or handle containing all y labels to names
            all_labels_file=self.all_labels_file        # full filepath or handle containing all labels to names
        )



class AlignedSimilarity:
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
        # tag = what distinguishes whether it's a probabilistic or deterministic embedding
        # and supervised or unsupervised
        self.dirname = dirname
        self.clsobj = clsobj
        self.seedvalues = seedvalues
        self.models = {}
        self.tag = '' if tag is None else f"{tag}_"

    def _load(self, seed):
        # load the model with the corresponding seed.
        # don't load all at once as this can lead to out of memory
        clsname = self.clsobj.__name__
        filename = os.path.join(self.dirname, clsname, f'{self.tag}{clsname}_{seed}.pt')
        model = torch.load(filename)
        return model

    def means(self, kernel, outfile_root, mode='a', mask=None):
        # kernel: callable that takes 2 arrays and returns similarity matrix
        # outfile: target file name, will have audioset or openimages appended to the start
        #          and should be a full path.
        # mask: if passed, should be a sequence of integers for which the
        # similarity will be calculated. It is up to the user to keep track
        # of what the final output indices mean.
        # Similarity will be calculated for each seed and stored in `outfile`
        clsname = self.clsobj.__name__

        for seed in self.seedvalues:
            model = self._load(seed)
            # we have to do both audioset and openimages.
            # openimages are the x-embeddings.
            # audioset are the y-embeddings.
            lookup = dict(
                openimages=model.aligner.x_emb.weight.detach().cpu().numpy(),
                audioset=model.aligner.y_emb.weight.detach().cpu().numpy()
            )
            for domain, values in lookup.items():
                dirname = os.path.join(os.path.dirname(outfile_root), domain)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                filename = os.path.basename(outfile_root)
                outfile = os.path.join(dirname, clsname,  f'{self.tag}{filename}')
                store = pd.HDFStore(outfile, mode=mode)
                thisvalue = kernel(values, values)
                store[str(seed)] = pd.DataFrame(thisvalue)
                store.close()

if __name__ == '__main__':

    import os
    import socket
    import kernels
    import gc
    remote = socket.gethostname().endswith('pals.ucl.ac.uk')
    if remote:
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

    epochs = 150
    probabilistic = True
    supervised = True

    seeds = (1,
             2, #3, 4, 5, 6, 7, 8, 9, 10
             )

    # Set to True if you want to calculate similarity of means
    analyse = True
    # Set to True only if you want the models to be saved with
    # increasing accuracy after 90%
    save = True
    mmds = [0,
            100]
    results_path = os.path.join(ppath, 'spond', 'experimental', 'glove', 'results')

    for mmd, seed in itertools.product(mmds, seeds):
        continue
        print("----------------------------------")
        print(f"Starting mmd {mmd}, seed {seed}")
        trainer = pl.Trainer(gpus=int(gpu), max_epochs=epochs, progress_bar_refresh_rate=20)
        # batch sizes larger than 100 causes a strange CUDA error with pytorch 1.7
        # Had to upgrade to pytorch 1.9
        # It may be due to some internal array being larger than 65535 when cdist is used.
        # https://github.com/pytorch/pytorch/issues/49928
        # https://discuss.pytorch.org/t/cuda-invalid-configuration-error-on-gpu-only/50399/15
        batch_size = 1000   # needs to be 500 for probabilistic, otherwise 1000 OOM
        y_cooc_file = os.path.join(datapath, 'audioset', "co_occurrence_audio_all.pt")
        y_labels_file = os.path.join(datapath, 'audioset', "class_labels.csv")
        y_dim = 6
        if remote:
            x_cooc_file = os.path.join(datapath, 'openimages', "co_occurrence.pt")
            x_labels_file = os.path.join(datapath, 'openimages', "oidv6-class-descriptions.csv")
            x_dim = 6
            all_labels_file = os.path.join(datapath, "all_labels.csv")
        else:
            # train audioset against itself
            x_cooc_file = y_cooc_file
            x_labels_file = y_labels_file
            x_dim = y_dim
            all_labels_file = x_labels_file

        datadict = DataDictionary(
            x_cooc=torch.load(x_cooc_file),
            x_labels_file=x_labels_file,
            y_cooc=torch.load(y_cooc_file),
            y_labels_file=y_labels_file,
            all_labels_file=all_labels_file,
            intersection_plus=None
        )
        model = AlignedGlove(batch_size,
                             data=datadict,
                             x_embedding_dim=x_dim,  # dimension of x
                             y_embedding_dim=y_dim,  # dimension of y
                             seed=seed,
                             probabilistic=probabilistic,
                             supervised=supervised, mmd=mmd,
                             save_flag=save,
                             # This is where the similarity class will look for output
                             outdir=os.path.join(results_path, 'AlignedGlove'),
                             max_epochs=epochs)    # don't like this duplication but we need it for the filename
        trainer.fit(model)
        print(f"Finished mmd {mmd}, seed {seed}: {model.filename}")
        del model
        del trainer
        del datadict
        torch.cuda.empty_cache()
        gc.collect()

    if analyse:
        for mmd in mmds:
            tag = f'probabilistic_sup_mmd{mmd}_150'
            print(f"Processing {tag}")
            sim = AlignedSimilarity(
                results_path, AlignedGlove, seedvalues=seeds, tag=tag)
            sim.means(kernels.dot, os.path.join(results_path, 'means_dot.hdf5'), mask=None, mode='a')
            del sim
            gc.collect()
