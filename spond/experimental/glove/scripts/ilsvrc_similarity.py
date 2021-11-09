# Run similarity comparisons with ILSVRC dataset.
# This was obtained from Brett.
# All of the data files should be in /home/petra/data/ilsvrc
# If they are not, they can be obtained from Brett

# This script should be run from the interactive shell-
# it sets variables which can then be inspected, dumped to tables, etc

import os
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


def load_ilsvrc(names_fn, data_fn):
    # get the data out of the files into a machine-readable structure
    names = pd.read_csv(names_fn, header=None, index_col=0).values.squeeze()
    N = len(names)
    data = np.zeros((N, N))
    for i, line in enumerate(open(data_fn, 'r')):
        items = line.strip().split(" ")
        data[i] = np.array([float(item) for item in items])
    # Square DF whose index and columns are the same.
    # This is a redundant structure but it's not very big
    # and this makes writing / reading easier.
    return pd.DataFrame(data=data, index=names, columns=names)


def compare_with_ilsvrc(aligned, openimages, audioset, ilsvrc_df, logtag=""):
    # Run comparison with HSJ given an AlignedGlove model and
    # 2 independently learnt ProbabilisticGlove models for openimages
    # which also hold the co-occurrences.
    # a pair can be tested if it is present in ILSVRC and in a co-occurrence matrix
    # with a score of non zero.

    datadict = aligned.data
    logtag = f"{logtag}:" if logtag else ""

    openimages_index_to_name = {v: datadict.x_names[k] for k, v in datadict.x_labels.items()}
    audioset_index_to_name = {v: datadict.y_names[k] for k, v in datadict.y_labels.items()}

    openimages_name_to_index = {v: k for k, v in openimages_index_to_name.items()}
    audioset_name_to_index = {v: k for k, v in audioset_index_to_name.items()}

    indpt = {'openimages': openimages, 'audioset': audioset}
    embs = {'openimages': aligned.aligner.x_emb, 'audioset': aligned.aligner.y_emb}
    names = {'openimages': openimages_index_to_name, 'audioset': audioset_index_to_name}
    indexes = {'openimages': openimages_name_to_index, 'audioset': audioset_name_to_index}
    #out = {'openimages': openimages_pairs, 'audioset': audioset_pairs}
    intersections = {'openimages': [], 'audioset': []}
    aligned_pairs = {'openimages': [], 'audioset': []}
    indpt_pairs = {'openimages': [], 'audioset': []}
    ilsvrc_pairs = {'openimages': [], 'audioset': []}
    rcorrs = {'openimages': {}, 'audioset': {}}
    domain_inds = {'openimages': [], 'audioset': []}

    ilsvrc_names = ilsvrc_df.index
    N = len(ilsvrc_names)

    for domain, name2index in [
        ('openimages', openimages_name_to_index),
        ('audioset', audioset_name_to_index)
    ]:
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

        # build up the list of items so we don't have to index later
        ilsvrc_domain = []
        counter = 0
        for i in range(N):
            name1 = ilsvrc_names[i]
            if domain == 'audioset' and name1 == 'tick':
                # we know that this one is a wrong mapping,
                # because the 'tick' referred to in ImageNet is the insect
                continue
            # Munge the ILSVRC names to be the same as the domain names
            n1 = name1.replace("_", " ")
            n1 = n1[0].upper() + n1[1:]
            for j in range(i+1, N):
                score = ilsvrc_df.values[i, j]
                name2 = ilsvrc_names[j]
                # convert _ to space, and uppercase the first letter.
                n2 = name2.replace("_",  " ")
                n2 = n2[0].upper() + n2[1:]
                ind1 = name2index.get(n1)
                ind2 = name2index.get(n2)
                # it may be present in ILSVRC but not in this domain
                if ind1 is None or ind2 is None:
                    continue

                # check if it occurred at all- a concept can be present in the
                # domain concept universe, but maybe it occurred with freq = 0
                if emb.coo_dense[ind1, ind2] == 0:
                    continue
                # otherwise just build the domain scores
                ilsvrc_domain.append(score)

                domain_inds[domain].append((ind1, ind2))
                intersections[domain].append((name1, name2))
                if counter % 1000 == 0:
                    print(f"{logtag}{domain}: {counter} pairs")
                counter += 1
        print(f"{logtag}{domain}: Final {counter} pairs")

        inds = np.array(domain_inds[domain]).T
        aligned_domain = embdist[inds[0], inds[1]]
        indpt_domain = inddist[inds[0], inds[1]]

        ilsvrc_pairs[domain] = np.array(ilsvrc_domain)
        aligned_pairs[domain] = np.array(aligned_domain)
        indpt_pairs[domain] = np.array(indpt_domain)
        # Done building indexes, now calculate correlations
        for embtype, values in [('aligned', aligned_pairs[domain]),
                                ('independent', indpt_pairs[domain])]:
            rcorr = stats.spearmanr(
                values,
                ilsvrc_pairs[domain]
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
        # spearman correlation of ILSVRC similarity with the domain similarity
        rcorrs=rcorrs,
        # intersection of audioset and ILSVRC concepts
        audioset_intersection=audioset_intersection,
        # intersection of openimages and ILSVRC concepts
        openimages_intersection=openimages_intersection
    )


if __name__ == '__main__':
    # set up pythonpath
    ppath = '/home/petra/spond'
    # set up data pth
    datapath = '/home/petra/data'
    resultspath = os.path.join(
        ppath, 'spond', 'experimental', 'glove', '../results')
    gpu = True

    sys.path.append(ppath)
    # Set this flag if you want to do the calculations
    # If set to False, then the results will be processed from the file.
    run = True
    seeds = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    mmds = [0, 100]
    rcorrs = {mmd: {} for mmd in mmds}
    accs = {mmd: {} for mmd in mmds}

    #ipth = os.path.join(ppath, 'spond', 'experimental', 'glove')
    ipth = os.path.join(datapath, 'ilsvrc')
    ilsvrc_df = load_ilsvrc(os.path.join(ipth, 'class_names.txt'),
                            os.path.join(ipth, 'marg_smat_4-221-6.txt'))
    ret = {}
    outfile = os.path.join(resultspath, 'probabilistic_sup_ilsvrc_similarity_cosine.pt')
    #ret = torch.load(os.path.join(resultspath, 'probabilistic_sup_ilsvrc_similarity_cosine.pt'))
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

            out = compare_with_ilsvrc(model, imgs, audio, ilsvrc_df, logtag=f"{seed}:mmd={mmd}")
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
        ret.update(dict(rcorrs=rcorrs, accs=accs))
        torch.save(ret, os.path.join(resultspath, outfile))
    else:
        ret = torch.load(os.path.join(resultspath, outfile))

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
