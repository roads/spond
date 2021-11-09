# Run similarity comparisons with WordNet.
# The nltk library will be required for this.

# This script should be run from the interactive shell-
# it sets variables which can then be inspected, dumped to tables, etc

from nltk.corpus import wordnet as wn
import nltk
import itertools
import os
import socket
import gc
import sys
import torch
from scipy import stats
from scipy.spatial import distance

import itertools
import pandas as pd
import numpy as np

from spond.experimental.glove.aligned_glove import AlignedGlove
from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove


def compare_with_wordnet(aligned, openimages, audioset, wordnet_pairs=None, logtag=""):
    # aligned: AlignedGlove model instance
    # openimages, audioset: ProbabilisticGlove model instances for the two domains
    # wordnet_pairs: pd.Series of wordnet similarities if pre-calculated, otherwise None
    #                will force this algorithm to construct it
    # logtag: String that will be used to tag logging messages
    #
    # Run comparison with WordNet simlarity given an AlignedGlove model and
    # 2 independently learnt ProbabilisticGlove models (for openimages and audioset)
    # which also hold the co-occurrences.
    # a pair can be tested if it is present in WordNet and in a co-occurrence matrix
    # more than once.
    # It takes a long time to run the wordnet_pairs, and they are the same for all embeddings
    # so allow passing it in with the wordnet_pairs argument.
    # If wordnet_pairs is not passed, this function will build the
    # data structure of pairs and then return it.
    #
    # We do not do any stemming of the concepts.
    # for now we will have to do an exact string match
    build_wn = wordnet_pairs is None
    if build_wn:
        # keys: (word1, word2)
        wordnet_pairs = {}
    else:
        print("Wordnet pairs already built")

    datadict = aligned.data
    logtag = f"{logtag}:" if logtag else ""

    openimages_index_to_name = {v: datadict.x_names[k] for k, v in datadict.x_labels.items()}
    audioset_index_to_name = {v: datadict.y_names[k] for k, v in datadict.y_labels.items()}

    openimages_name_to_index = {v: k for k, v in openimages_index_to_name.items()}
    audioset_name_to_index = {v: k for k, v in audioset_index_to_name.items()}

    indpt = {'openimages': openimages, 'audioset': audioset}
    embs = {'openimages': aligned.aligner.x_emb, 'audioset': aligned.aligner.y_emb}
    names = {'openimages': openimages_index_to_name, 'audioset': audioset_index_to_name}
    #indexes = {'openimages': openimages_name_to_index, 'audioset': audioset_name_to_index}
    intersections = {'openimages': [], 'audioset': []}
    aligned_pairs = {'openimages': [], 'audioset': []}
    indpt_pairs = {'openimages': [], 'audioset': []}
    wn_pairs = {'openimages': [], 'audioset': []}
    rcorrs = {'openimages': {}, 'audioset': {}}
    domain_inds = {'openimages': [], 'audioset': []}
    if build_wn:
        for domain in ('openimages', 'audioset'):
            emb = embs[domain]
            # emb.co_occurrence is a sparse matrix, so calling indices() on it
            # will return pairs that occur nonzero times.
            # Iterate over each pair of indices. The indices will be used
            # to get the concept names.
            for i, (ind1, ind2) in enumerate(emb.co_occurrence.indices().t().cpu().numpy()):
                if i % 1000 == 0:
                    print(f"{logtag}{domain}: i={i}")
                # the concepts are guaranteed to be present by construction
                d1 = names[domain].get(ind1)
                d2 = names[domain].get(ind2)
                # prepare for passing to wordnet by converting to the same form
                # all lower case, with spaces replaced by underscores.
                word1 = d1.lower().replace(" ", "_")
                word2 = d2.lower().replace(" ", "_")
                # if either word is not in wordnet, stop.
                # This is just a plain text match with no stemming
                ws1 = wn.synsets(word1)
                if not ws1:
                    continue
                ws2 = wn.synsets(word2)
                if not ws2:
                    continue
                # wordnet synsets have the same word in different senses.
                # we don't know which sense to take, so we will take all the
                # similarity combinations of the first 2 synsets in each.
                # Example of where this works better:
                # mandarin_orange in Wordnet has 2 senses: The tree, and the fruit
                # comparing mandarin_orange(tree) with orange has v low similarity
                # comparing mandarin(fruit) with orange has high similarity
                # so this heuristic should hopefully pick up the greatest similarity
                sims = []
                # ws1[:2] will return the first 2 if there are >=2 or the first 1 only
                for syn1, syn2 in itertools.product(ws1[:2], ws2[:2]):
                    if syn1._pos != syn2._pos:
                        # must have same part of speech, or else
                        # similarity is not defined.
                        continue
                    sim = syn1.lch_similarity(syn2)
                    sims.append(sim)
                # in case somehow all combinations are different parts of speech
                # then we have nothing to check.
                if not sims:
                    continue
                # Just take the maximum - so this algorithm will be biased high.
                score = np.max(sims)
                wordnet_pairs[(d1, d2)] = score
        # Finally build the series of pairs and their similarity
        # as measured by WordNet LCH similarity.
        # Index is multi-index of word pairs.
        wordnet_pairs = pd.Series(wordnet_pairs)
    # iterate over wordnet_pairs for each domain.
    for domain, name2index in [('openimages', openimages_name_to_index),
                               ('audioset', audioset_name_to_index)]:
        print(f"{logtag}Processing {domain}")
        emb = embs[domain]
        embweight = emb.weight.detach().cpu().numpy()
        indweight = indpt[domain].glove_layer.weight.detach().cpu().numpy()
        # Calculate all the cosine similarity at once.
        # distance.cdist gives DISTANCE, so we have to convert to similarity
        # so that the correlation between LCH similarity and this will be appropriate.
        embdist = 1 - distance.cdist(embweight, embweight, metric='cosine')
        inddist = 1 - distance.cdist(indweight, indweight, metric='cosine')
        # build up the list of items so we don't have to index later
        wn_domain = []
        counter = 0
        for (name1, name2), score in wordnet_pairs.items():
            ind1 = name2index.get(name1)
            ind2 = name2index.get(name2)
            # it may be present in wordnet but not in this domain
            if ind1 is None or ind2 is None:
                continue
            # check if it occurred at all: 0 indicates it did not occur
            if emb.coo_dense[ind1, ind2] == 0:
                continue
            # otherwise just build the domain scores
            wn_domain.append(score)
            # Store the indices as a large array of pairs
            # we will use this for numpy fancy indexing later.
            domain_inds[domain].append((ind1, ind2))
            intersections[domain].append((name1, name2))
            if counter % 1000 == 0:
                print(f"{logtag}{domain}: {counter} pairs")
            counter += 1
        # Done. Build the structures.
        inds = np.array(domain_inds[domain]).T
        # embdist/inddist are 2D numpy arrays. Use the inds for fancy indexing
        # to get everything at once.
        aligned_domain = embdist[inds[0], inds[1]]
        indpt_domain = inddist[inds[0], inds[1]]

        wn_pairs[domain] = np.array(wn_domain)
        aligned_pairs[domain] = np.array(aligned_domain)
        indpt_pairs[domain] = np.array(indpt_domain)
        # Done building indexes, now calculate correlations
        for embtype, values in [('aligned', aligned_pairs[domain]),
                                ('independent', indpt_pairs[domain])]:
            rcorr = stats.spearmanr(
                values,
                wn_pairs[domain]
            )
            rcorrs[domain][embtype] = rcorr[0]
            rcorrs[domain][f"{embtype}_p"] = rcorr[1]
            print(f"{logtag}{domain} {embtype}: rcorr={rcorr[0]} with p={rcorr[1]}")
        del embdist
        del inddist
        gc.collect()

    rcorrs = pd.DataFrame(rcorrs)
    audioset_intersection = pd.Index(intersections['audioset'])
    openimages_intersection = pd.Index(intersections['openimages'])
    return dict(
        rcorrs=rcorrs,
        wordnet_pairs=wordnet_pairs,
        audioset_intersection=audioset_intersection,
        openimages_intersection=openimages_intersection
    )


if __name__ == '__main__':
    # set up pythonpath
    ppath = '/home/petra/spond'
    # set up data pth
    datapath = '/home/petra/data'
    resultspath = os.path.join(
        ppath, 'spond', 'experimental', 'glove', 'results')
    gpu = True

    sys.path.append(ppath)
    # Set this flag if you only want to read pre-processed results.
    run = False
    seeds = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    mmds = [0, 100]
    rcorrs = {mmd: {} for mmd in mmds}
    accs = {mmd: {} for mmd in mmds}
    # Store for existing wordnet pairs. If this is the first run, comment these out
    store = pd.HDFStore(os.path.join(resultspath, 'AlignedGlove', 'wordnet_pairs.hdf5'), mode='r')
    wn_pairs = store['wordnet_pairs']
    store.close()
    ret = {}
    if run:
        for seed, mmd in itertools.product(seeds, mmds):
            print(f"Processing seed={seed}, mmd={mmd}")
            modelfile = os.path.join(resultspath, 'AlignedGlove', f'probabilistic_sup_mmd{mmd}_150_AlignedGlove_{seed}.pt')

            model = AlignedGlove.load(modelfile)
            accs[mmd].setdefault('openimages', {})
            accs[mmd].setdefault('audioset', {})
            accs[mmd]['openimages'][seed] = model.losses.acc_x.iloc[-1]
            accs[mmd]['audioset'][seed] = model.losses.acc_y.iloc[-1]

            imgs = ProbabilisticGlove.load(
                os.path.join(resultspath, 'openimages', 'ProbabilisticGlove', f'openimages_ProbabilisticGlove_{seed}.pt')
            )

            audio = ProbabilisticGlove.load(
                os.path.join(resultspath, 'audioset', 'ProbabilisticGlove', f'audioset_ProbabilisticGlove_{seed}.pt')
            )

            out = compare_with_wordnet(model, imgs, audio, wordnet_pairs=wn_pairs, logtag=f"{seed}:mmd={mmd}")
            if wn_pairs is None:
                wn_pairs = out['wordnet_pairs']
            rcorrs[mmd][seed] = out.pop('rcorrs')
            if ret.get('audioset_intersection') is None:
                ret['audioset_intersection'] = out.pop('audioset_intersection')
            if ret.get('openimages_intersection') is None:
                ret['openimages_intersection'] = out.pop('openimages_intersection')
            del model
            del imgs
            del audio
            del out
            gc.collect()
        ret.update(dict(rcorrs=rcorrs, accs=accs, wordnet_pairs=wn_pairs))
        torch.save(ret, os.path.join(resultspath, 'probabilistic_sup_wordnet_similarity_cosine.pt'))
    else:
        ret = torch.load(os.path.join(resultspath, 'probabilistic_sup_wordnet_similarity_cosine.pt'))

    accs = ret['accs']

    def calc_accs(mmd, domain):
        allaccs = pd.Series(accs[mmd][domain])
        return allaccs

    rcorrs = ret['rcorrs']

    def calc_diffs(mmd, domain):
        diffs = []
        for seed in range(1, 11):
            baseline = rcorrs[mmd][seed][domain]['independent']
            aligned = rcorrs[mmd][seed][domain]['aligned']
            diffs.append(aligned-baseline)
        return np.mean(diffs)

    def rcorrs_df(domain):
        rcorrs_0 = pd.DataFrame({
            seed: rcorrs[0][seed][domain].loc[['aligned', 'independent']]
            for seed in range(1, 11)
        })
        rcorrs_100 = pd.DataFrame({
            seed: rcorrs[100][seed][domain].loc[['aligned', 'independent']]
            for seed in range(1, 11)
        })
        df = rcorrs_0.T
        df['aligned_mmd'] = rcorrs_100.T['aligned']
        df = df[['independent', 'aligned', 'aligned_mmd']]
        return df

    cols = ['independent', 'aligned', 'aligned_acc', 'aligned_mmd', 'aligned_mmd_acc']

    img_stats = rcorrs_df('openimages')
    img_stats['aligned_acc'] = calc_accs(0, 'openimages')
    img_stats['aligned_mmd_acc'] = calc_accs(100, 'openimages')
    img_stats = img_stats.T
    img_stats['mean'] = img_stats.mean(axis=1)
    img_stats = img_stats.T
    img_stats = img_stats[cols]
    img_diffs = pd.DataFrame({
        'aligned': img_stats['aligned'] - img_stats['independent'],
        'aligned_mmd': img_stats['aligned_mmd'] - img_stats['independent'],
    })

    audio_stats = rcorrs_df('audioset')
    audio_stats['aligned_acc'] = calc_accs(0, 'audioset')
    audio_stats['aligned_mmd_acc'] = calc_accs(100, 'audioset')
    audio_stats = audio_stats.T
    audio_stats['mean'] = audio_stats.mean(axis=1)
    audio_stats = audio_stats.T
    audio_stats = audio_stats[cols]
    audio_diffs = pd.DataFrame({
        'aligned': audio_stats['aligned'] - audio_stats['independent'],
        'aligned_mmd': audio_stats['aligned_mmd'] - audio_stats['independent'],
    })
    # At this point the img_* and audio_* variables can be inspected
    # from the interactive shell.
