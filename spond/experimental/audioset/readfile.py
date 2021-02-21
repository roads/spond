# This module processes AudioSet data files
# https://research.google.com/audioset/download.html
from copy import copy

import csv
import torch
import numpy as np
import os

from spond.experimental.openimage.readfile import readlabels as readlabels_imgs


def combine_files(imagefn, audiofn, outputfn, rootdir="."):
    # The audio and image file tags are the same
    # as in, the same entity has the same machine ID in both files
    # Therefore, to be consistent, we take all the audio tags
    # that are not in the image tag set, and combine them
    # to make a third output file.
    # This way, the image co-occurrence does not have to be re-processed.
    _, labels_to_names = readlabels_imgs(imagefn, rootdir=None)
    #labels_to_names = {v: k for k, v in names_openimage.items()}
    # the audio file is in a different format
    # index,mid,display_name
    # 0,/m/09x0r,"Speech"
    # 1,/m/05zppz,"Male speech, man speaking"
    # 2,/m/02zsn,"Female speech, woman speaking"
    # Read the audio input and concatenate with the image input
    # into this format:
    #
    # #LabelName,DisplayName
    # /m/011k07,Tortoise
    # /m/011q46kg,Container
    # /m/012074,Magpie
    with open(os.path.join(rootdir, audiofn)) as fh:
        # format is:
        #index,mid,display_name
        #0,/m/09x0r,"Speech"
        #1,/m/05zppz,"Male speech, man speaking"
        csvreader = csv.reader(fh, delimiter=',', quotechar='"')
        for idx, tokens in enumerate(csvreader):
            #tokens = line.split(",")
            # if the first token isn't an int, skip the whole line
            try:
                int(tokens[0])
            except:
                continue
            # strip the quotes
            label, name = tokens[1], tokens[2]
            # only add it if it's not already present
            if label not in labels_to_names:
                labels_to_names[label] = name
    with open(os.path.join(rootdir, outputfn), 'w') as fh:
        fh.write("#LabelName,DisplayName\n")
        for label, name in labels_to_names.items():
            fh.write(f"{label},{name}\n")
    labels_to_idx, labels_to_names = readlabels_imgs(outputfn)
    return labels_to_idx, labels_to_names


def generate_cooccurrence(segments_filename, labels_to_idx, rootdir='.'):
    fn = os.path.join(rootdir, segments_filename)
    # The audio file is not large, so no need to multiprocess
    # keys: image index, values: {label index, score}
    segmentlabels = {}
    # keys: [(label index, other label index)], values: score
    coo = {}
    # need 2 passes through the file, one is to get all the segment IDs
    # and map them to indexes
    # we don't have URLs to these easily so for the moment ignore any mapping
    segment_to_idx = {}
    with open(fn, 'r') as fh:
        idx = 0
        for line in fh:
            if line.startswith("#"):
                continue
            tokens = line.strip().split(",")
            segment = tokens[0].strip()
            segment_to_idx[segment] = idx
            idx += 1
    allsegments = []
    with open(fn, 'r') as fh:
        csvreader = csv.reader(fh, delimiter=',', quotechar='"')
        for vals in csvreader:
            # File format is:
            """
            # Segments csv created Sun Mar  5 10:54:31 2017
            # num_ytids=22160, num_segs=22160, num_unique_labels=527, num_positive_labels=52882
            # YTID, start_seconds, end_seconds, positive_labels
            --PJHxphWEs, 30.000, 40.000, "/m/09x0r,/t/dd00088"
            --ZhevVpy1s, 50.000, 60.000, "/m/012xff"
            --aE2O5G5WE, 0.000, 10.000, "/m/03fwl,/m/04rlf,/m/09x0r"
            --aO5cdqSAg, 30.000, 40.000, "/t/dd00003,/t/dd00005"
            """
            if vals[0].startswith("#"):
                continue
            # fields we care about for now are:
            # ImageID (position 0)
            # LabelName (position 2)
            # Confidence (position 3)
            segmentname = vals[0]
            allsegments.append(segmentname)
            labels = vals[3:]
            # there is potentially more than 1 label
            segmentidx = segment_to_idx[segmentname]
            # keep track of what other labels are in this segment
            segmentlabels.setdefault(segmentidx, {})
            for label in labels:
                # labels may be padded with spaces or have a quote character
                label = label.strip().replace('"', "").strip()
                labelidx = labels_to_idx[label]
                segmentlabels[segmentidx].setdefault(labelidx, 0)
                score = 1    # if it was found in the file at all
                segmentlabels[segmentidx][labelidx] += score
                for otherlabel in segmentlabels[segmentidx]:
                    if labelidx == otherlabel:
                        continue
                    coo.setdefault((labelidx, otherlabel), 0)
                    coo.setdefault((otherlabel, labelidx), 0)
                    coo[(labelidx, otherlabel)] += score
                    coo[(otherlabel, labelidx)] += score
    sorted_keys = sorted(coo.keys())
    data = np.fromiter((coo[k] for k in sorted_keys),
                       dtype=float, count=len(sorted_keys))
    skarr = np.array(sorted_keys).T
    i = torch.LongTensor(skarr)
    v = torch.FloatTensor(data)
    nlabels = len(labels_to_idx)
    coo_torch = torch.sparse.FloatTensor(i, v, (nlabels, nlabels))
    return coo_torch


if __name__ == '__main__':
    #rootdir = "/home/petra/data"
    rootdir = '.'

    # Download from
    # http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
    # and merge with
    # https://storage.googleapis.com/openimages/v5/class-descriptions.csv
    labels_to_idx, labels_to_names = combine_files(
        "../openimage/oidv6-class-descriptions.csv",
        "./class_labels_indices.csv",
        "./all_labels.csv"
    )

    # all_train_segments.csv is a combination of
    # http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv
    # and
    # http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv

    allfn = 'all_train_segments.csv'
    coo_pt = generate_cooccurrence(allfn, labels_to_idx, rootdir=rootdir)
    coo_pt = coo_pt.coalesce()
    torch.save(coo_pt, 'co_occurrence_audio_all.pt')
