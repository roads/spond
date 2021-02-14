import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class GloveDataset:

    def init__(self, text, n_words=200000, window_size=5):
        self._window_size = window_size
        self._tokens = text.split(" ")[:n_words]
        word_counter = Counter()
        word_counter.update(self._tokens)
        # our equivalent: labels to index
        self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
        self._id2word = {i:w for w, i in self._word2id.items()}
        # our equivalent: total number of labels
        self._vocab_len = len(self._word2id)

        self._id_tokens = [self._word2id[w] for w in self._tokens]

        self._create_coocurrence_matrix()

        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)
        # equivalent: label IDs
        self._i_idx = list()
        # equivalent: label IDs
        self._j_idx = list()
        # equivalent: values
        self._xij = list()

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx)#.cuda()
        self._j_idx = torch.LongTensor(self._j_idx)#.cuda()
        self._xij = torch.FloatTensor(self._xij)#.cuda()

    def __init__(self, filename):
        # Load the pre-constructed co_occurrence.pt which should be a
        # sparse tensor.
        self.cooc_mat = torch.load(filename).coalesce()
        # get into the right shape for batch generation
        self.indices = self.cooc_mat.indices().t()
        self.values = self.cooc_mat.values()
        self.concept_len = self.indices.max().item() + 1

    def get_batches(self, batch_size):
        N = self.concept_len
        rand_ids = torch.LongTensor(np.random.choice(N, N, replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            indices = self.indices[[batch_ids]]
            yield self.values[[batch_ids]], indices[:, 0], indices[:, 1]


#rootdir = '/opt/github.com/spond/spond/experimental/openimage'
rootdir = '/home/petra/spond/spond/experimental/openimage'


dataset = GloveDataset(os.path.join(rootdir, 'co_occurrence.pt'))


EMBED_DIM = 300
class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.wi = nn.Embedding(num_embeddings, embedding_dim)
        self.wj = nn.Embedding(num_embeddings, embedding_dim)
        self.bi = nn.Embedding(num_embeddings, 1)
        self.bj = nn.Embedding(num_embeddings, 1)

        self.wi.weight.data.uniform_(-1, 1)
        self.wj.weight.data.uniform_(-1, 1)
        self.bi.weight.data.zero_()
        self.bj.weight.data.zero_()

    def forward(self, i_indices, j_indices):
        w_i = self.wi(i_indices)
        w_j = self.wj(j_indices)
        b_i = self.bi(i_indices).squeeze()
        b_j = self.bj(j_indices).squeeze()

        x = torch.sum(w_i * w_j, dim=1) + b_i + b_j

        return x


glove = GloveModel(dataset.concept_len, EMBED_DIM)

TRAIN = True
PLOT = True


def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx#.cuda()

def wmse_loss(weights, inputs, targets):
    loss = weights * F.mse_loss(inputs, targets, reduction='none')
    return torch.mean(loss)#.cuda()

if TRAIN:

    optimizer = optim.Adagrad(glove.parameters(), lr=0.05)
    #optimizer = optim.Adam(glove.parameters(), lr=0.001)

    N_EPOCHS = 500
    BATCH_SIZE = 2048
    X_MAX = 100
    ALPHA = 0.75
    n_batches = int(dataset.concept_len / BATCH_SIZE)
    loss_values = list()
    min_loss = np.inf
    l = np.inf
    for e in range(1, N_EPOCHS+1):
        batch_i = 0

        for x_ij, i_idx, j_idx in dataset.get_batches(BATCH_SIZE):

            batch_i += 1

            optimizer.zero_grad()

            outputs = glove(i_idx, j_idx)
            weights_x = weight_func(x_ij, X_MAX, ALPHA)
            loss = wmse_loss(weights_x, outputs, torch.log(x_ij))

            loss.backward()

            optimizer.step()
            l = loss.item()
            loss_values.append(l)

            #if batch_i % 1024 == 0:
            print("Epoch: {}/{} \t Batch: {}/{} \t Loss: {}".format(e, N_EPOCHS, batch_i, n_batches, np.mean(loss_values[-20:])))
        print("Saving model...")
        if l < min_loss:
            min_loss = l
            torch.save(glove.state_dict(), "glove_min_500.pt")
        torch.save(glove.state_dict(), "glove.pt")

else:

    glove.load_state_dict(torch.load('glove_min_500.pt'))

if PLOT:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # Download from https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv
    labelsfn = 'oidv6-class-descriptions.csv'

    # Download from https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv
    imgfn = 'oidv6-train-images-with-labels-with-rotation.csv'


    import sys

    sys.path.append('/home/petra/spond/spond/experimental')
    
    from openimage.readfile import readlabels, readimgs

    labels, names = readlabels(labelsfn, rootdir='/home/petra/data')
    idx_to_name = {
        v: names[k] for k, v in labels.items()
    }

    emb_i = glove.wi.weight.data.numpy()
    emb_j = glove.wj.weight.data.numpy()
    emb = emb_i + emb_j
    top_k = 300
    tsne = TSNE(metric='cosine', n_components=2, random_state=123, init='pca', perplexity=100.0, n_iter=5000)

    # find the most commonly co-occuring items
    # These are the items which appear the most times
    incidences = dataset.cooc_mat.to_dense().sum(axis=0)
    indexes = np.argsort(incidences)
    top_k_indices = indexes[-top_k:]

    #embed_tsne = tsne.fit_transform(emb[:top_k, :])
    embed_tsne = tsne.fit_transform(emb[top_k_indices, :])
    fig = plt.figure(figsize=(50, 50))
    #ax = fig.add_subplot(111, projection='3d')

    for idx, concept_idx in enumerate(top_k_indices):
        m = embed_tsne[idx, :]
        plt.scatter(*m, color='steelblue')
        concept = idx_to_name[concept_idx.item()]
        plt.annotate(concept, (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
        #ax.text(m[0], m[1], m[2],  concept, size=20, zorder=1,
        #        color='k')

    plt.savefig('glove_500_iters.png')
