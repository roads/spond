# We want to take 100 samples of the following embeddings for each domain:
#
# ProbabilisticGlove
# AlignedGlove without MMD
# AlignedGlove with MMD
#
# Take the sample mean, and then take the correlation of the sample means
# with various things.

import gc
import os
import sys
import itertools
import pandas as pd

import numpy as np
import torch
from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove
from spond.experimental.glove.aligned_glove import AlignedGlove

basepath = "/home/petra/spond/spond/experimental/glove"

resultspath = os.path.join(basepath, "results")

models = {
    "AlignedGlove": {
        # call the lambda function with a seed to get the specific model instance
        0: (lambda seed: AlignedGlove.load(
            os.path.join(resultspath, "AlignedGlove", f"probabilistic_sup_mmd0_150_AlignedGlove_{seed}.pt"))),
        100: (lambda seed: AlignedGlove.load(
            os.path.join(resultspath, "AlignedGlove", f"probabilistic_sup_mmd100_150_AlignedGlove_{seed}.pt"))),
    },
    "ProbabilisticGlove": {
        0: {
            "openimages": (lambda seed: ProbabilisticGlove.load(
                os.path.join(resultspath, "openimages", "ProbabilisticGlove",
                             f"openimages_ProbabilisticGlove_{seed}.pt"))),
            "audioset": (lambda seed: ProbabilisticGlove.load(
                os.path.join(resultspath, "audioset", "ProbabilisticGlove",
                             f"audioset_ProbabilisticGlove_{seed}.pt"))),
        }
    }
}


# map domain to name of attribute of AlignedGlove.aligner
attrnames = {"openimages": "x_emb", "audioset": "y_emb"}

# number of samples to take
N = 100

seeds = np.arange(1, 11)

# Set this flag to actually take the samples and save to a file
# then set to False and rerun
run = False


if not run:
    store = pd.HDFStore(os.path.join(resultspath, "samples_similarity.hdf5"), "r")
    out = pd.HDFStore(os.path.join(resultspath, "samples_dotsim_crosscorr.hdf5"))

    for domain in ('openimages', 'audioset'):
        for tag in ('AlignedGlove_0', 'AlignedGlove_100', 'ProbabilisticGlove'):
            crosscorrs = {}
            full = f"{domain}_{tag}"
            print(f"{domain}:{full}")
            data = store[full]
            for i, seed1 in enumerate(seeds):
                for seed2 in seeds[i+1:]:
                    print(f"{seed1} x {seed2}")
                    crosscorr = np.corrcoef(data[seed1].values, data[seed2].values)[0][1]
                    crosscorrs[(seed1, seed2)] = crosscorr
            out[full] = pd.Series(crosscorrs)
            out.flush()
            del crosscorrs
            del data
            gc.collect()

    store.close()
else:
    sims = {
        'openimages': {
            'AlignedGlove': {0: {seed: None for seed in seeds},
                             100: {seed: None for seed in seeds}},
            'ProbabilisticGlove': {0: {seed: None for seed in seeds}}
        },
        'audioset': {
            'AlignedGlove': {0: {seed: None for seed in seeds},
                             100: {seed: None for seed in seeds}},
            'ProbabilisticGlove': {0: {seed: None for seed in seeds}}
        },
    }

    for seed in seeds:
        print(f"Seed {seed}")
        for domain in ("openimages", "audioset"):
            print(f"Calculating independent {domain}")
            ind = models["ProbabilisticGlove"][0][domain](seed)
            ind_emb = ind.glove_layer.weights(n=N)
            ind_emb = ind_emb.cuda()
            # take the mean of the samples
            ind_sim = torch.einsum('...ij,...kj->ik', ind_emb, ind_emb)/N
            sims[domain]['ProbabilisticGlove'][0][seed] = ind_sim.detach().cpu().numpy().ravel()
            del ind_emb
            del ind_sim
            del ind
            torch.cuda.empty_cache()
            gc.collect()

        for mmd in (0, 100):
            model = models["AlignedGlove"][mmd](seed)
            aligner = model.aligner
            for domain in ("openimages", "audioset"):
                print(f"Calculating aligned {domain}, mmd={mmd}")
                aligned = getattr(aligner, attrnames[domain])
                aligned_emb = aligned.weights(n=N)
                aligned_emb = aligned_emb.cuda()
                aligned_sim = torch.einsum('...ij,...kj->ik', aligned_emb, aligned_emb)/N
                sims[domain]['AlignedGlove'][mmd][seed] = aligned_sim.detach().cpu().numpy().ravel()
                del aligned_emb
                del aligned_sim
                torch.cuda.empty_cache()
                gc.collect()
            del aligner
            del model
            torch.cuda.empty_cache()
            gc.collect()
    store = pd.HDFStore(os.path.join(resultspath, "samples_similarity.hdf5"))
    # Do not remove any of the del / gc calls.
    # Each dataframe is quite big and thus we only allow one to be in memory
    # at any given time.
    store['openimages_AlignedGlove_0'] = pd.DataFrame(sims['openimages']['AlignedGlove'][0])
    del sims['openimages']['AlignedGlove'][0]
    gc.collect()
    store['openimages_AlignedGlove_100'] = pd.DataFrame(sims['openimages']['AlignedGlove'][100])
    del sims['openimages']['AlignedGlove'][100]
    gc.collect()
    store['openimages_ProbabilisticGlove'] = pd.DataFrame(sims['openimages']['ProbabilisticGlove'][0])
    del sims['openimages']['ProbabilisticGlove']
    gc.collect()

    store['audioset_AlignedGlove_0'] = pd.DataFrame(sims['audioset']['AlignedGlove'][0])
    del sims['audioset']['AlignedGlove'][0]
    gc.collect()
    store['audioset_AlignedGlove_100'] = pd.DataFrame(sims['audioset']['AlignedGlove'][100])
    del sims['audioset']['AlignedGlove'][100]
    gc.collect()
    store['audioset_ProbabilisticGlove'] = pd.DataFrame(sims['audioset']['ProbabilisticGlove'][0])
    del sims['audioset']['ProbabilisticGlove']
    gc.collect()
    store.close()


