from collections import Counter, defaultdict

import numpy as np
import torch



class GloveWordsDataset:
    # TODO: Need to refactor so that all datasets take a co-occurrence matrix
    # instead of building in here
    def __init__(self, text, n_words=200000, window_size=5, device='cpu'):
        self._window_size = window_size
        self._tokens = text.split(" ")[:n_words]
        word_counter = Counter()
        word_counter.update(self._tokens)
        self._word2id = {w:i for i, (w,_) in enumerate(word_counter.most_common())}
        self._id2word = {i:w for w, i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        self.concept_len = self._vocab_len

        self._id_tokens = [self._word2id[w] for w in self._tokens]
        self.device = device

        self._create_coocurrence_matrix()
        print("# of words: {}".format(len(self._tokens)))
        print("Vocabulary length: {}".format(self._vocab_len))

    def _create_coocurrence_matrix(self):
        device = self.device
        cooc_mat = defaultdict(Counter)
        for i, w in enumerate(self._id_tokens):
            start_i = max(i - self._window_size, 0)
            end_i = min(i + self._window_size + 1, len(self._id_tokens))
            for j in range(start_i, end_i):
                if i != j:
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1 / abs(j-i)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx).to(device)
        self._j_idx = torch.LongTensor(self._j_idx).to(device)
        self._xij = torch.FloatTensor(self._xij).to(device)
        self.N = len(self._i_idx)

    def get_batches(self, batch_size):
        #Generate random idx
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


class GloveDataset:

    def __init__(self, filename, device='cpu'):
        # Load the pre-constructed co_occurrence.pt which should be a
        # sparse tensor.
        self.device = device
        self.cooc_mat = torch.load(filename).coalesce().to(device)
        # get into the right shape for batch generation
        self.indices = self.cooc_mat.indices().t().to(device)
        self.values = self.cooc_mat.values().to(device)
        self.concept_len = self.indices.max().item() + 1
        self.N = self.indices.shape[0]

    def get_batches(self, batch_size):
        N = self.N
        rand_ids = torch.LongTensor(np.random.choice(N, N, replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            indices = self.indices[[batch_ids]]
            yield self.values[[batch_ids]], indices[:, 0], indices[:, 1]
