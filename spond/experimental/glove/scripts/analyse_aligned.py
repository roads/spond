import gc
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy import stats
import itertools

import socket
remote = socket.gethostname().endswith('pals.ucl.ac.uk')
if remote:
    # set up pythonpath
    ppath = '/home/petra/spond'
    # set up data path
    datapath = '/home/petra/data'
    gpu = False
    resultspath = '/home/petra/spond/spond/experimental/glove/results'
else:
    1/0
    ppath = '/opt/github.com/spond/spond/experimental'
    datapath = ppath
    gpu = False
    tag = 'audioset'
    labelsfn = "/opt/github.com/spond/spond/experimental/audioset/all_labels.csv"
    resultspath = '/opt/github.com/spond/spond/experimental/glove/results/'
    train_cooccurrence_file = ''

device = torch.device("cuda:0" if gpu else "cpu")

sys.path.append(ppath)
sys.path.append(os.path.join(ppath, 'spond', 'experimental', 'glove'))

# Can only import after path is set above
from spond.experimental.glove.aligned_glove import AlignedGlove, DataDictionary
from spond.experimental.openimages.readfile import readlabels

y_cooc_file = os.path.join(datapath, 'audioset', "co_occurrence_audio_all.pt")
y_labels_file = os.path.join(datapath, 'audioset', "class_labels.csv")
y_dim = 6
x_cooc_file = os.path.join(datapath, 'openimages', "co_occurrence.pt")
x_labels_file = os.path.join(datapath, 'openimages', "oidv6-class-descriptions.csv")
x_dim = 6
all_labels_file = os.path.join(datapath, "all_labels.csv")

datadict = DataDictionary(
    x_cooc=torch.load(x_cooc_file),
    x_labels_file=x_labels_file,
    y_cooc=torch.load(y_cooc_file),
    y_labels_file=y_labels_file,
    all_labels_file=all_labels_file,
    intersection_plus=None
)

domains = ('audioset', 'openimages', )


regs = [
    #"",
    #"0.001", "0.01",
    #"0.1",
    #"0.5", #still running
    #"1.0"
]

import itertools

mmds = (0, 100)

for domain, mmd in itertools.product(domains, mmds):
    tag = f'probabilistic_sup_mmd{mmd}_150'
    print("-----------------------------------_")
    print(f"{domain}: Processing {tag}")
    # label to name
    included_labels_dict = dict(
        openimages=pd.DataFrame({
            'mid': pd.Series({v: k for k, v in datadict.x_labels.items()}),
            'display_name': pd.Series({v: datadict.x_names[k] for k, v in datadict.x_labels.items()}),
        }),
        audioset=pd.DataFrame({
            'mid': pd.Series({v: k for k, v in datadict.y_labels.items()}),
            'display_name': pd.Series({v: datadict.y_names[k] for k, v  in datadict.y_labels.items()}),
        })
    )

    cooc_dict = dict(
        openimages=x_cooc_file,
        audioset=y_cooc_file
    )

    rdir = os.path.join(resultspath, domain, 'AlignedGlove')


    #s = pd.HDFStore(os.path.join(rdir, f'{tag}_means_dot.hdf5'), 'r')
    s = pd.HDFStore(os.path.join(rdir, f'{tag}_means_cosine.hdf5'), 'r')
    seeds = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    for seed in seeds:
        continue
        df = s[str(seed)]
        print(f"Calculating self-correlation for seed {seed}")
        corrs = np.corrcoef(df.values)
        plt.figure()
        fig = plt.imshow(corrs)
        plt.colorbar(fig)
        plt.title(f'Correlation of dot product similarity, {seed}\n{tag}')
        plt.savefig(os.path.join(rdir, f'{tag}_dotsim_corr_{seed}.png'))
        plt.close()
        del fig
        gc.collect()

        # print(f"Calculating self-distance for seed {seed}")
        # dist = cdist(df.values, df.values)  # euclidean is the default
        # plt.figure()
        # fig = plt.imshow(dist)
        # plt.colorbar(fig)
        # plt.title(f'Distance of dot product similarity, {seed}')
        # plt.savefig(os.path.join(rdir, f'{tag}_dotsim_dist_{seed}.png'))
        # del fig
        del corrs
        del df
        gc.collect()

    # now work out cross correlations

    outfile = pd.HDFStore(os.path.join(rdir, f'{tag}_analytics.hdf5'), 'a')
    
    crosscorrs = {}

    for i, seed1 in enumerate(seeds):
        for seed2 in seeds[i+1:]:
            print(f"Calculating cosine cross-correlation for {seed1} x {seed2}")
            c1 = s[str(seed1)].values.ravel()
            c2 = s[str(seed2)].values.ravel()
            crosscorrs[(seed1, seed2)] = np.corrcoef(c1, c2)[0][1]
            del c1
            del c2
            gc.collect()

    s.close()

    crosscorrs = pd.Series(crosscorrs)
    outfile['crosscorrs_cosine'] = crosscorrs
    outfile.flush()

    del crosscorrs
    gc.collect()
    continue
    included_labels = included_labels_dict[domain]

    metrics = ('distance', 'correlation')

    raw_results_path = '/home/petra/spond/spond/experimental/glove/results/AlignedGlove'

    entropies = {}

    # map which domain goes to which embedding
    attrnames = dict(
        openimages='x_emb',
        audioset='y_emb',
    )

    for metric in metrics:
        models = {}
        model_means = {}
        most = {}
        least = {}
        corrmeans = {}

        for seed in seeds:
            print(f"Calculating max/min {metric}s for seed {seed}")
            model = AlignedGlove.load(os.path.join(raw_results_path, f'{tag}_AlignedGlove_{seed}.pt'))

            if domain == 'audioset':
                emb = model.aligner.y_emb
            else:
                emb = model.aligner.x_emb

            #models[seed] = model
            attrname = attrnames[domain]
            model_means [seed] = getattr(model.aligner, attrname).wi_mu.weight.detach().cpu().numpy()
            # the code in the branches below is intentionally duplicated sometimes,
            # because we have to garbage collect to avoid big structures being
            # retained in memory after they are not needed.
            if metric == 'correlation':
                cc = np.corrcoef(emb.wi_mu.weight.detach().numpy())
                # replace values of 1 with inf or -inf so that we can sort easily
                mostlike = cc.copy()
                mostlike[np.isclose(cc, 1)] = -np.inf
                # top are the indexes of the highest correlations sorted by column
                # do .copy() here because later in the loop we can then delete
                # mostlike and leastlike, and free a lot of memory
                top = mostlike.argsort(axis=0)[-5:][::-1].copy()
                maxes = pd.Series({
                    included_labels['display_name'][i]:
                        pd.Series(index=included_labels['display_name'][top[:, i]].values,
                                  data=mostlike[i][top[:, i]])
                    for i in range(mostlike.shape[0])
                })
                del mostlike
                gc.collect()

                leastlike = cc.copy()
                leastlike[np.isclose(cc, 1)] = np.inf
                bottom = leastlike.argsort(axis=0)[:5].copy()

                mins = pd.Series({
                    included_labels['display_name'][i]:
                        pd.Series(index=included_labels['display_name'][bottom[:, i]].values,
                                  data=leastlike[i][bottom[:, i]])
                    for i in range(leastlike.shape[0])
                })
                del leastlike
                del cc
                gc.collect()
            else:
                assert metric == 'distance'
                wt = emb.wi_mu.weight.detach()
                wt = wt.to(device)
                # compute_mode="donot_use_mm_for_euclid_dist" is required or else
                # the distance between something and itself is not 0
                # don't know why.
                dist = torch.cdist(wt, wt, compute_mode="donot_use_mm_for_euclid_dist")
                dist = dist.detach().cpu()
                # same as above, replace values of 0 with inf or -inf so we can sort
                mostlike = dist.clone()
                mostlike[torch.isclose(dist, torch.tensor([0.0]))] = np.inf
                mostlike = mostlike.numpy()
                top = mostlike.argsort(axis=0)[:5].copy()
                # make into data structures
                maxes = pd.Series({
                    included_labels['display_name'][i]:
                        pd.Series(index=included_labels['display_name'][top[:, i]].values,
                                  data=mostlike[i][top[:, i]])
                    for i in range(wt.shape[0])
                })

                del mostlike
                torch.cuda.empty_cache()
                gc.collect()

                leastlike = dist.clone()
                leastlike[torch.isclose(dist, torch.tensor([0.0]))] = -np.inf
                # top are the indexes of the lowest distances sorted by column
                # see note in the other branch about copy() and memory management
                leastlike = leastlike.cpu().numpy()

                bottom = leastlike.argsort(axis=0)[-5:][::-1].copy()
                mins = pd.Series({
                    included_labels['display_name'][i]:
                        pd.Series(index=included_labels['display_name'][bottom[:, i]].values,
                                  data=leastlike[i][bottom[:, i]])
                    for i in range(wt.shape[0])
                })
                del leastlike
                del dist
                torch.cuda.empty_cache()
                gc.collect()

            most[seed] = maxes
            least[seed] = mins

            ## Need to make this work
            # calculate entropy
            ent = emb.entropy().detach()
            # sort it
            ents, indices = ent.sort()
            ordered_labels = [included_labels['display_name'][item] for item in indices.numpy()]
            entropies[seed] = pd.Series(
                data=ents.numpy().copy(), index=ordered_labels
            )
            # delete everything not needed so we can free memory for the next loop
            del top
            del bottom
            del ent
            del model
            gc.collect()

        most = pd.DataFrame(most)
        least = pd.DataFrame(least)
        outfile[f'mostalike_{metric}'] = most
        outfile[f'leastalike_{metric}'] = least

        del most
        del least
        gc.collect()

    entropies = pd.Series(entropies)

    plt.figure()
    for seed in seeds:
        plt.hist(entropies[seed].values, alpha=0.3, bins=100, label=str(seed))
    plt.title(f'Entropies for {tag} per seed')
    plt.legend()
    plt.savefig(os.path.join(rdir, f'{tag}_entropies.png'))
    plt.close()
    outfile['entropies'] = entropies

    # calculate correlations of counts with entropies, for each seed
    # entropies index are alphabetical, we have to match up with the counts

    cooc = torch.load(cooc_dict[domain])
    cooc = cooc.coalesce().to_dense()
    counts = cooc.sum(axis=0).numpy()
    counts = pd.Series(data=counts)

    # we want to compare with the single modality deterministic
    single_modality_fn = os.path.join(datapath, domain, f'glove_{domain}.pt')

    deterministic = torch.load(single_modality_fn)

    det_embeddings = deterministic['wi.weight'].data + deterministic['wj.weight'].data

    det_embeddings = det_embeddings.detach().cpu()

    name_to_index_map = dict(
        openimages={v: datadict.x_labels[k] for k, v in datadict.x_names.items()},
        audioset={v: datadict.y_labels[k] for k, v in datadict.y_names.items()},
    )

    name_to_index = name_to_index_map[domain]

    attrname = attrnames[domain]

    det_learnt_corr = pd.Series({
        # calculate correlation between deterministic and learnt means
        seed: np.corrcoef(
            #getattr(models[seed].aligner, attrname).wi_mu.weight.detach().cpu().numpy(),
            model_means[seed],
            det_embeddings.numpy()
        )[0][1]
        for seed in seeds
    })

    outfile['det_learnt_corr'] = det_learnt_corr

    # Pearson correlation to check for linear relationship
    entropy_count_corr = pd.Series({
        seed: np.corrcoef(
            counts.loc[[name_to_index[ind] for ind in entropies[seed].index]],
            entropies[seed].values
        )[0][1] for seed in seeds
    })

    outfile['entropy_count_corr'] = entropy_count_corr

    # Spearman correlation to check for monotonic relationship
    entropy_count_rcorr = {'spearmanr': {}, 'p': {}}

    for seed in seeds:
        rc, p = stats.spearmanr(
            counts.loc[[name_to_index[ind] for ind in entropies[seed].index]],
            entropies[seed].values
        )
        entropy_count_rcorr['spearmanr'][seed] = rc
        entropy_count_rcorr['p'][seed] = p

    entropy_count_rcorr = pd.DataFrame(entropy_count_rcorr)

    outfile['entropy_count_rcorr'] = entropy_count_rcorr

    outfile.close()
    del models
    gc.collect()
