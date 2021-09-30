# This module contains useful stuff to generate various plots and analyses.
# analyse_aligned.py and analyse.py must be run before this,
# in order to generate the HDF5 files that contain the data.
import itertools
import gc
import numpy as np
import os
import pandas as pd
import sys
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

# https://adjusttext.readthedocs.io/en/latest/index.html
# Needed only to make the labels pretty
from adjustText import adjust_text

# set up data path
resultspath = "/home/petra/spond/spond/experimental/glove/results"
datapath = "/home/petra/data"
sys.path.append('/home/petra/spond')
sys.path.append('/home/petra/spond/spond/experimental/glove')

# Set this flag if you want to load data from all the results into memory.
# It takes a while
load = True

from spond.experimental.openimages.readfile import readlabels
from spond.experimental.glove.aligned_glove import AlignedGlove
from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove

all_labels_file = os.path.join(datapath, 'all_labels.csv')

# labels = machine IDs to index
# names = machine IDs to names
all_labels, all_names = readlabels(all_labels_file, rootdir=None)

name_to_label = {v: k for k, v in all_names.items()}
index_to_label = {v: k for k, v in all_labels.items()}
index_to_name = {v: all_names[k] for k, v in all_labels.items()}
name_to_index = {v: k for k, v in index_to_name.items()}

# each of the labels files is just a big list of machine IDs and their
# corresponding names
datafiles = {
    'openimages': {
        'labels':  os.path.join(datapath, 'openimages', 'oidv6-class-descriptions.csv'),
    },
    'audioset': {
        'labels':  os.path.join(datapath, 'audioset', 'class_labels.csv'),
    },
}

lookup = {
    'openimages': {}, 'audioset': {},
}


domains = ("openimages",
           "audioset",)

mmds = [0, 100]
tags = [f'probabilistic_sup_mmd{mmd}_150' for mmd in mmds]

for domain in domains:
    labelsfn = datafiles[domain]['labels']
    included_labels = pd.read_csv(os.path.join(datapath, domain, labelsfn),
                                  header=0)
    datafiles[domain]['included_labels'] = included_labels
    # for consistency. The openimages one has different titles
    included_labels.columns = ['mid', 'display_name']
    keep = np.array([all_labels[label] for label in included_labels['mid'].values])
    lookup[domain]['included_index'] = keep
    lookup[domain]['included_names'] = [index_to_name[ind] for ind in keep]
    # we also need the index of the name in this domain
    lookup[domain]['name_to_index'] = {
        dd['display_name']: idx for idx, dd in included_labels.iterrows()
    }

# now find the labels that are in both domains, and the indexes of those labels
# in the embeddings.
union = [
    item for item in lookup['audioset']['included_names']
    if item in lookup['openimages']['included_names']
]

for domain in domains:
    lookup[domain]['union'] = {
        name: lookup[domain]['name_to_index'][name] for name in union
    }

stores = {"AlignedGlove": {}, "ProbabilisticGlove": {}}
corrs = {"AlignedGlove": {}, "ProbabilisticGlove": {}}
dists = {"AlignedGlove": {}, "ProbabilisticGlove": {}}
# If the "load" flag is set, load correlations, distances, models
# into memory so we can call various functions from interactive interpreter
for tag, mmd in zip(tags, mmds):
    if not load:
        continue
    model_name = "AlignedGlove"
    t_stores = {
        domain: pd.HDFStore(os.path.join(resultspath, domain, model_name, f"{tag}_analytics.hdf5"), 'r')
        for domain in domains
    }

    t_corrs = {
        domain: t_stores[domain]['mostalike_correlation']
        for domain in domains
    }

    t_dists = {
        domain: t_stores[domain]['mostalike_distance']
        for domain in domains
    }

    stores[model_name][mmd] = t_stores
    corrs[model_name][mmd] = t_corrs
    dists[model_name][mmd] = t_dists
if load:

    t_stores =  {
        domain: pd.HDFStore(os.path.join(resultspath, domain, "ProbabilisticGlove", f"{domain}_analytics.hdf5"), 'r')
        for domain in domains
    }
    t_corrs = {
        domain: t_stores[domain]['mostalike_correlation']
        for domain in domains
    }
    t_dists = {
        domain: t_stores[domain]['mostalike_distance']
        for domain in domains
    }

    stores["ProbabilisticGlove"][0] = t_stores
    dists["ProbabilisticGlove"][0] = t_dists
    corrs["ProbabilisticGlove"][0] = t_corrs


def mostalike(domain, concept, metric, mmd, model_name="AlignedGlove", seeds=[1,2,3,4,5]):
    lookup = corrs if metric == 'correlation' else dists
    out = {}
    print(model_name)
    for seed in seeds:
        print(f"Seed: {seed}")
        print(f"Best by {metric}:")
        val = lookup[model_name][mmd][domain][seed][concept]
        print(val)
        out[seed] = pd.Series(index=np.arange(1, 6), data=list(val.items()))
    return pd.DataFrame(out)


def format_mostalike(df, seeds=[1,2,3,4,5]):
    N = len(seeds)
    prefix = r"""
    \begin{table}[]
    \begin{tabular}{@{}""" + ("l" * (N+1)) + """@{}}
Rank / Seed &""" + "&".join([f"{s} " for s in seeds]) + r"\\"
    out = [prefix]
    postfix = r"""
    \end{tabular}
    \end{table}
"""
    for i in np.arange(1, 6):
        thisrow = [str(i)]
        for seed in seeds:
            wrapper = r"\begin{{tabular}}[c]{{@{{}}l@{{}}}} {label} \\ {value:.3f} \end{{tabular}}"
            label, value = df[seed].loc[i]
            thisrow.append(wrapper.format(label=label, value=value))
        out.append("&".join(thisrow) + r" \\")

    out.append(postfix)
    return "\n".join(out)


def get_model_details(domain, tag, seed, model_name='AlignedGlove'):
    # given a model identifying details like name, domain, tag, seed
    # return the appropriate embeddings and co-occurrence that correspond to it.
    if model_name == "AlignedGlove":
        fn = os.path.join(
            resultspath, 'AlignedGlove', f"{tag}_AlignedGlove_{seed}.pt")
        model = AlignedGlove.load(fn)
    elif model_name == "ProbabilisticGlove":
        fn = os.path.join(
            resultspath, domain, 'ProbabilisticGlove', f"{domain}_ProbabilisticGlove_{seed}.pt")
        model = ProbabilisticGlove.load(fn)
    else:
        raise AssertionError(f"Invalid model name {model_name}")

    if model_name == "AlignedGlove":
        if domain == 'openimages':
            emb = model.aligner.x_emb
            cooc = model.data.x_cooc
            mapping_fn = model.aligner.fx
        else:
            emb = model.aligner.y_emb
            cooc = model.data.y_cooc
            mapping_fn = model.aligner.gy
    else:
        emb = model.glove_layer
        cooc_file = os.path.join(datapath, domain, 'co_occurrence.pt')
        cooc = torch.load(cooc_file)
        mapping_fn = None

    # the mapping for index_to_name depends on the domain
    if model_name == "AlignedGlove":
        mapped_indexes = model.data.x_indexes if domain == 'openimages' else model.data.y_indexes
        mapped_indexes = mapped_indexes.numpy()
        index_to_name = {
            i: model.data.index_to_name[ind] for i, ind in enumerate(mapped_indexes)
        }
    else:
        if domain == "openimages":
            labelsfn = os.path.join(datapath, domain, 'oidv6-class-descriptions.csv')
        else:
            labelsfn = os.path.join(datapath, domain, 'class_labels.csv')


        labels, names = readlabels(labelsfn, rootdir=None)

        name_to_label = {v: k for k, v in names.items()}
        index_to_label = {v: k for k, v in labels.items()}
        index_to_name = {v: names[k] for k, v in labels.items()}
        name_to_index = {v: k for k, v in index_to_name.items()}

    return model, emb, cooc, mapping_fn, index_to_name


def tsne(domain, tag, seed, intersection=True, top_k=200, model_name="AlignedGlove", adjust=True,
         mapped=False, figsize=(30, 30)):
    # if intersection is True: The intersection points will always be included.
    # If top_k is also true, the top K most common points will also be plotted.
    # If mapped is True, will plot the mapped version of the domain eg if domain=openimages, mapped=True,
    # the projection of openimages into audioset embedding space will be plotted.
    # load the model corresponding to the tag/seed and run TSNE on it
    # Different naming convention for aligned or probabilistic
    # things need to be saved in the right directory
    if model_name == "ProbabilisticGlove":
        assert top_k > 0, "ProbabilisticGlove must be used with top_k != 0"
    model, emb, cooc, mapping_fn, index_to_name = get_model_details(domain, tag, seed, model_name)
    if intersection and model_name == "AlignedGlove":
        if domain == 'openimages':
            intersect = model.data.intersection_indexes[:, 1]
        else:
            intersect = model.data.intersection_indexes[:, 2]
    else:
        intersect = []
    add_indexes = []
    if top_k:
        dense = cooc.to_dense().cpu()
        incidences = dense.sum(axis=0)
        nonzero = np.nonzero(incidences)
        nonzero_incidences = incidences[nonzero]
        add_indexes = np.argsort(nonzero_incidences.t()).squeeze()
        top_k = min(top_k, add_indexes.shape[0])
        add_indexes = nonzero[add_indexes[-top_k:]].t().squeeze()
    indexes = np.unique(np.hstack([intersect, add_indexes])).astype(int)

    print(indexes)
    # at this point, "indexes" gives the index into the co-occurrence
    # we need to map backwards to get the actual names

    tsne = TSNE(metric='cosine', n_components=2, random_state=123)
    input_emb = emb.weight.detach().cpu()
    if mapped:
        input_emb = mapping_fn.cpu()(input_emb).detach()
    embeddings = tsne.fit_transform(input_emb.numpy()[indexes, :])

    fig = plt.figure(figsize=figsize)

    texts = []

    for idx, concept_idx in enumerate(indexes):
        m = embeddings[idx, :]
        col = 'steelblue' if concept_idx in intersect else 'orange'
        plt.scatter(*m, color=col)
        concept = index_to_name[concept_idx.item()]
        txt = plt.annotate(concept, (embeddings[idx, 0], embeddings[idx, 1]),
                           alpha=0.7, fontsize=12)
        texts.append(txt)
    if adjust:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='green'))
    outfile = f"tsne_{domain}_{tag}_{model_name}_{seed}"
    if intersection:
        title2 = 'intersection'
        if top_k:
            title2 += f' plus top {top_k}'
            outfile = f"intersection_top{top_k}_{outfile}.png"
        else:
            outfile = f"intersection_{outfile}.png"
    else:
        outfile = f"top{top_k}_{outfile}.png"
        title2 = f'top {top_k}'
    if mapped:
        outfile = f"mapped_{outfile}"
    outdir = os.path.join(resultspath, model_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, outfile)
    domain = "" if not mapped else f"mapped({domain})"
    plt.title(f"{model_name}: {domain}: {tag}, seed={seed}\n{title2}")
    plt.savefig(outfile)
    plt.savefig(outfile)
    plt.close()
    del model
    gc.collect()



def tsne_both(tag, seed, top_k=200, adjust=True, figsize=(30, 30)):
    # Will plot TSNE of the embeddings in the shared transformed space
    # that is: f(x) and g(y). Will plot top_k plus intersection.

    model_name = "AlignedGlove"
    model = AlignedGlove.load(os.path.join(resultspath, model_name, f"{tag}_{model_name}_{seed}.pt"))

    #model, emb, cooc, index_to_name = get_model_details(domain, tag, seed, model_name)

    x_intersect = model.data.intersection_indexes[:, 1]
    y_intersect = model.data.intersection_indexes[:, 2]
    x_index_to_name =  {
        i: model.data.index_to_name[ind.item()] for i, ind in enumerate(model.data.x_indexes)
    }
    y_index_to_name =  {
        i: model.data.index_to_name[ind.item()] for i, ind in enumerate(model.data.y_indexes)
    }
    # keep track of which indexes are the ones to be plotted.
    indexes = {'openimages': None, 'audioset': None}
    for domain, intersect,  cooc in [
            ("openimages", x_intersect, model.data.x_cooc, ),
            ("audioset", y_intersect,  model.data.y_cooc, ),
    ]:

        add_indexes = []
        if top_k:
            dense = cooc.to_dense().cpu()
            incidences = dense.sum(axis=0)
            nonzero = np.nonzero(incidences)
            nonzero_incidences = incidences[nonzero]
            add_indexes = np.argsort(nonzero_incidences.t()).squeeze()
            top_k = min(top_k, add_indexes.shape[0])
            add_indexes = nonzero[add_indexes[-top_k:]].t().squeeze()
        indexes[domain] = np.unique(np.hstack([intersect, add_indexes])).astype(int)

    print(indexes)
    # at this point, "indexes" gives the index into the co-occurrence.
    # stick them together
    all_indexes = np.hstack([indexes['openimages'], indexes['audioset']])


    all_embs = np.vstack([
        model.aligner.fx(model.aligner.x_emb.weight).detach().cpu().numpy()[indexes['openimages'], :],
        model.aligner.gy(model.aligner.y_emb.weight).detach().cpu().numpy()[indexes['audioset'], :],
    ])
    tsne = TSNE(metric='cosine', n_components=2, random_state=123)
    embeddings = tsne.fit_transform(all_embs)

    fig = plt.figure(figsize=figsize)

    texts = []
    for domain, col, index_to_name in [
            ('openimages', 'red', x_index_to_name),
            ('audioset', 'green', y_index_to_name)
    ]:
        for idx, concept_idx in enumerate(indexes[domain]):
            if domain == 'audioset':
                idx += len(indexes['openimages'])
            m = embeddings[idx, :]
            plt.scatter(*m, color=col)
            concept = index_to_name[concept_idx.item()]
            txt = plt.annotate(concept, (embeddings[idx, 0], embeddings[idx, 1]),
                               alpha=0.7, fontsize=12)
            texts.append(txt)
    if adjust:
        adjust_text(texts, arrowprops=dict(arrowstyle='->', color='green'))
    outfile = f"tsne_both_{tag}_{model_name}_{seed}"
    outfile = f"top{top_k}_{outfile}.png"
    title2 = f'top {top_k}: red=Open Images, green=AudioSet'
    outdir = os.path.join(resultspath, model_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, outfile)
    plt.title(f"{model_name}: shared embedding space: {tag}, seed={seed}\n{title2}")
    plt.savefig(outfile)
    plt.savefig(outfile)
    plt.close()
    del model
    gc.collect()


def get_entropies(model_name, mmd, domain, concept, seeds=[1,2,3,4,5]):
    return pd.Series({
        seed: stores[model_name][mmd][domain]['entropies'][seed][concept]
        for seed in seeds
    })


def plot_entropies(ax, model_name, mmd, domain, seeds=[1,2,3,4,5,6,7,8,9,10]):
    print(f"Plotting {model_name}:{domain}:{mmd}")
    df = {}
    for seed in seeds:
        df[seed] = stores[model_name][mmd][domain]['entropies'][seed].sort_index()
    df = pd.DataFrame(df)
    # from https://stackoverflow.com/questions/33864578/matplotlib-making-labels-for-violin-plots
    vp = ax.violinplot(df.values)
    color = vp['bodies'][0].get_facecolor().flatten()
    return mpatches.Patch(color=color)


def plot_entropies_box(model_name, mmd, domain, seeds=[1,2,3,4,5,6,7,8,9,10]):
    print(f"Plotting {model_name}:{domain}:{mmd}")
    df = {}
    for seed in seeds:
        df[seed] = stores[model_name][mmd][domain]['entropies'][seed].sort_index()
    df = pd.DataFrame(df)
    plt.figure()
    plt.boxplot(df.values)
    plt.title(f"Box plot of entropies for {domain} per seed")
    plt.xlabel("seed")
    plt.ylabel("entropy")
    #color = vp['bodies'][0].get_facecolor().flatten()
    #return mpatches.Patch(color=color)
    outfile = os.path.join(resultspath, domain,  f'{domain}_entropies_box.png')
    plt.savefig(outfile)
    plt.close()


def plot_all_entropies(seeds=[1,2,3,4,5,6,7,8,9,10]):
    for domain in ('audioset', 'openimages'):
        outfile = os.path.join(resultspath, domain,  f'{domain}_entropies_violin.png')
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title(f"{domain}: \nDistribution of entropies for model variants per random seed")
        plt.xlabel("seed")
        plt.ylabel("entropy")

        labels = []
        for lb, model, mmd in [ ('independent', 'ProbabilisticGlove', 0),
                                ('aligned', 'AlignedGlove', 0),
                                ('aligned+MMD', 'AlignedGlove', 100),]:
            patch = plot_entropies(ax, model, mmd, domain)
            labels.append((patch, lb))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(*zip(*labels), loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(outfile)
        plt.close()


def aligned_vs_ind_corrs(mmd, domain, seeds=[1,2,3,4,5,6,7,8,9,10]):
    ind = pd.HDFStore(
        os.path.join(resultspath, domain, 'ProbabilisticGlove', f"{domain}_means_dot.hdf5"),
        "r"
    )
    aligned = pd.HDFStore(
        os.path.join(resultspath, domain, 'AlignedGlove', f"probabilistic_sup_mmd{mmd}_150_means_dot.hdf5"),
        "r"
    )
    corrs = {}
    for seed in seeds:
        print(f"Calculating correlations between aligned / independent for {domain}:{mmd}:{seed}")
        ind_sim = ind[str(seed)]
        aligned_sim = aligned[str(seed)]
        corrs[seed] = np.corrcoef(ind_sim.values.ravel(), aligned_sim.values.ravel())[0][1]
        del ind_sim
        del aligned_sim
        gc.collect()
    ind.close()
    aligned.close()
    return pd.Series(corrs)


def stability(dists, model_name, mmd, domain, seeds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], top_k=300,
              cooc=None, index_to_name=None, method='intersection'):
    # If method is intersection:
    # count the number of concepts intersecting in 5 nearest neighbours, divide by 5.
    # If method is union:
    # count the union of 5 nearest neighbours of each concept, divide by 5, take reciprocal.

    # to store the indexes of the top k concepts, only needs to be calculated once
    indexes  = None
    assert method in ("intersection", "union")
    d = dists[model_name][mmd][domain]
    # ordered by most to least frequent
    stability = {}
    tag = f"probabilistic_sup_mmd{mmd}_150"
    if cooc is None:
        _, cooc, _, index_to_name = get_model_details(domain, tag, 1, model_name)
    dense = cooc.to_dense().cpu()
    incidences = dense.sum(axis=0)
    nonzero = np.nonzero(incidences)
    nonzero_incidences = incidences[nonzero]
    indexes = np.argsort(nonzero_incidences.t()).squeeze()
    indexes = nonzero[indexes[-top_k:]].t().squeeze()
    indexes = indexes.int()

    for idx in reversed(indexes):
        if method == 'intersection':
            labels = None
        else:
            labels = pd.Index([])
        name = index_to_name[idx.item()]
        print(f"Counting stability for {tag}:{name}")
        nns = d.loc[name]
        for seed in seeds:
            if method == 'intersection':
                if labels is None:
                    labels = nns[seed].index
                else:
                    labels = labels.intersection(nns[seed].index)
            else:
                labels = labels.union(nns[seed].index)
        stability[name] = len(labels)/5 if method == 'intersection' else 5/len(labels)
    return pd.Series(stability)

def calc_stabilities():

    stabilities = {}

    for mmd, domain in itertools.product(mmds, domains):
        stabilities[('AlignedGlove', mmd, domain)] = stability('AlignedGlove', mmd, domain)


    for domain in domains:
        stabilities[('ProbabilisticGlove', 0, domain)] = stability('ProbabilisticGlove', 0, domain)


    torch.save(stabilities, os.path.join(resultspath, 'stabilities.pt'))

    sdf = pd.DataFrame(stabilities)

    dd = stabilities

    plt.figure()

    plt.plot(dd[('AlignedGlove', 0, 'audioset')].values, label='aligned, without MMD', linestyle='None', marker='o', alpha=0.6)
    plt.plot(dd[('AlignedGlove', 100, 'audioset')].values, label='aligned, with MMD', linestyle='None', marker='o',alpha=0.6)
    plt.plot(dd[('ProbabilisticGlove', 0, 'audioset')].values, label='independent', linestyle='None', marker='o',alpha=0.6)
    plt.title('AudioSet stability for top 300 concepts\n plotted in order of most frequent concept to least frequent')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(os.path.join(resultspath, 'audioset_stability.png'))
    plt.close()


    plt.figure();
    plt.plot(dd[('AlignedGlove', 0, 'openimages')].values, label='aligned, without MMD', linestyle='None', marker='o',alpha=0.6)
    plt.plot(dd[('AlignedGlove', 100, 'openimages')].values, label='aligned, with MMD', linestyle='None', marker='o',alpha=0.6)
    plt.plot(dd[('ProbabilisticGlove', 0, 'openimages')].values, label='independent', linestyle='None', marker='o',alpha=0.6)
    plt.title('Open Images stability for top 300 concepts\n plotted in order of most frequent concept to least frequent')
    plt.grid()
    plt.legend(loc='best')
    plt.savefig(os.path.join(resultspath, 'openimages_stability.png'))
    plt.close()

#if __name__ == '__main__':
#    import itertools

    #plot_entropies_box('ProbabilisticGlove', 0, 'openimages')
    #plot_entropies_box('ProbabilisticGlove', 0, 'audioset')

    #plot_all_entropies(seeds)
    #outfile = pd.HDFStore(os.path.join(resultspath, 'aligned_vs_ind_corrs.hdf5'))
    #for mmd, domain in itertools.product([#0,
    #                                      100], ["openimages"]): #domains):
    #    corrs = aligned_vs_ind_corrs(mmd, domain)
    #    outfile[f"{domain}_mmd{mmd}"] = corrs
    #outfile.close()
    #sys.exit()
    #tsne('openimages', tag, seed, intersection=True, top_k=300, model_name="AlignedGlove", adjust=True,
    #         figsize=(20,  20))
    #    tsne('audioset', tag, seed, intersection=True, top_k=300, model_name="AlignedGlove", adjust=True,
    #         figsize=(20, 20))
