import itertools
import os
import socket
import gc
import sys
import torch
from scipy import stats
from scipy.spatial import distance
import pandas as pd
import numpy as np

from spond.experimental.glove.aligned_glove import AlignedGlove
from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove


def compare_with_hsj(aligned, openimages, audioset):
    # Run comparison with HSJ given an AlignedGlove model and
    # 2 independently learnt ProbabilisticGlove models for openimages and audioset

    hsj = pd.read_csv(os.path.join(datapath, 'mturk-771.csv'), index_col=(1, 2))
    # a pair can be tested if it is present in HSJ and in a co-occurrence matrix
    # with a score of non zero.
    # for now we will have to do an exact string match
    # iterate over all the hsj word1 and word2 pairs and assume that
    # keys for all: word1, word2
    hsj_pairs = {}
    openimages_pairs = {'aligned': {}, 'independent': {}}
    audioset_pairs = {'aligned': {}, 'independent': {}}

    datadict = aligned.data

    openimages_name_to_index = {v.lower(): datadict.x_labels[k] for k, v in datadict.x_names.items()}
    audioset_name_to_index = {v.lower(): datadict.y_labels[k] for k, v in datadict.y_names.items()}
    counter = 0
    for (word1, word2), series in hsj.iterrows():
        score = series['similarity']

        hsj_pairs[(word1, word2)] = score

        # check openimages
        img1 = openimages_name_to_index.get(word1)
        img2 = openimages_name_to_index.get(word2)

        if img1 is not None and img2 is not None:
            for embtype, emb in [('aligned', aligned.aligner.x_emb),
                                 ('independent', openimages.glove_layer)]:
                img1_emb = emb.weight[img1].detach().cpu().numpy()
                img2_emb = emb.weight[img2].detach().cpu().numpy()
                # use dot product distance
                #img_score = (img1_emb * img2_emb).sum()
                img_score = 1 - distance.cosine(img1_emb, img2_emb)
                openimages_pairs[embtype][(word1, word2)] = img_score

        # check audioset
        audio1 = audioset_name_to_index.get(word1)
        audio2 = audioset_name_to_index.get(word2)
        if audio1 is not None and audio2 is not None:
            for embtype, emb in [('aligned', aligned.aligner.y_emb),
                                 ('independent', audioset.glove_layer)]:

                audio1_emb = emb.weight[audio1].detach().cpu().numpy()
                audio2_emb = emb.weight[audio2].detach().cpu().numpy()
                # use dot product distance
                # audio_score = (audio1_emb * audio2_emb).sum()
                audio_score = 1 - distance.cosine(audio1_emb, audio2_emb)
                audioset_pairs[embtype][(word1, word2)] = audio_score
        counter += 1
        if counter % 100 == 0:
            print(f"counter={counter}")
    hsj_pairs = pd.Series(hsj_pairs)
    openimages_pairs = pd.DataFrame(openimages_pairs)
    audioset_pairs = pd.DataFrame(audioset_pairs)
    openimages_ixn = openimages_pairs.index.intersection(hsj_pairs.index)
    audioset_ixn = audioset_pairs.index.intersection(hsj_pairs.index)
    rcorrs = {'openimages': {}, 'audioset': {}}
    for embtype in ('aligned', 'independent'):
        openimages_rcorr = stats.spearmanr(
            openimages_pairs[embtype].loc[openimages_ixn].values,
            hsj_pairs.loc[openimages_ixn].values
        )
        rcorrs['openimages'][embtype] = openimages_rcorr[0]
        rcorrs['openimages'][f"{embtype}_p"] = openimages_rcorr[1]
        audioset_rcorr = stats.spearmanr(
            audioset_pairs[embtype].loc[audioset_ixn].values,
            hsj_pairs.loc[audioset_ixn].values
        )
        rcorrs['audioset'][embtype] = audioset_rcorr[0]
        rcorrs['audioset'][f"{embtype}_p"] = audioset_rcorr[1]

    rcorrs = pd.DataFrame(rcorrs)
    return dict(
        rcorrs=rcorrs,
        hsj_pairs=hsj_pairs,
        audioset_pairs=audioset_pairs,
        openimages_pairs=openimages_pairs
    )



if __name__ == '__main__':
    remote = socket.gethostname().endswith('pals.ucl.ac.uk')
    if remote:
        # set up pythonpath
        ppath = '/home/petra/spond'
        # set up data pth
        datapath = '/home/petra/data'
        resultspath = os.path.join(ppath, 'spond', 'experimental', 'glove',
                                   '../results')
        gpu = True
    else:
        ppath = '/opt/github.com/spond/spond/experimental'
        #datapath = ppath
        datapath = '/home/petra/data'

        gpu = False

    sys.path.append(ppath)

    seeds = (1, 2, 3, 4, 5,  6,  7, 8, 9, 10)

    mmds = [0, 100]
    results = {mmd: {} for mmd in mmds}
    accs = {mmd: {} for mmd in mmds}
    for seed, mmd in itertools.product(seeds, mmds):
        print(f"{seed}:mmd={mmd}")
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

        out = compare_with_hsj(model, imgs, audio)
        results[mmd][seed] = out.pop('rcorrs')
        del model
        del imgs
        del audio
        gc.collect()
    ret = dict(results=results, accs=accs)
    ret.update(out)
    torch.save(ret, os.path.join(resultspath, 'probabilistic_sup_hsj_similarity_cosine.pt'))



    def calc_accs(accs, mmd, domain):
        allaccs = pd.Series(accs[mmd][domain])
        return allaccs

    def calc_diffs(rcorrs, mmd, domain):
        diffs = []
        for seed in range(1, 11):
            baseline = rcorrs[mmd][seed][domain]['independent']
            aligned = rcorrs[mmd][seed][domain]['aligned']
            diffs.append(aligned-baseline)
        return np.mean(diffs)

    def rcorrs_df(rcorrs, domain):
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

    accs = ret['accs']

    rcorrs = ret['results']

    cols = ['independent', 'aligned', 'aligned_acc', 'aligned_mmd', 'aligned_mmd_acc']

    img_stats = rcorrs_df(rcorrs, 'openimages')
    img_stats['aligned_acc'] = calc_accs(accs, 0, 'openimages')
    img_stats['aligned_mmd_acc'] = calc_accs(accs, 100, 'openimages')
    img_stats = img_stats.T
    img_stats['mean'] = img_stats.mean(axis=1)
    img_stats = img_stats.T
    img_stats = img_stats[cols]
    img_diffs = pd.DataFrame({
        'aligned': img_stats['aligned'] - img_stats['independent'],
        'aligned_mmd': img_stats['aligned_mmd'] - img_stats['independent'],
    })

    audio_stats = rcorrs_df(rcorrs, 'audioset')
    audio_stats['aligned_acc'] = calc_accs(accs, 0, 'audioset')
    audio_stats['aligned_mmd_acc'] = calc_accs(accs, 100, 'audioset')
    audio_stats = audio_stats.T
    audio_stats['mean'] = audio_stats.mean(axis=1)
    audio_stats = audio_stats.T
    audio_stats = audio_stats[cols]
    audio_diffs = pd.DataFrame({
        'aligned': audio_stats['aligned'] - audio_stats['independent'],
        'aligned_mmd': audio_stats['aligned_mmd'] - audio_stats['independent'],
    })
