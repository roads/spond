# all in one script for now, until we figure out what the real workflow should be
import torch
import numpy as np
import time
import subprocess
import itertools
import scipy.sparse
import os


def timed(f):
    # This decorator runs the decorated function and reports
    # how long it took
    def f_(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        total = end-start
        print(f"Ran {f.__name__} in {total} s")
        return ret

    return f_



@timed
def readlabels(labelsfn, rootdir='.'):
    # load the labels into a big dictionary: keys = labels, values = index
    # TODO: How will we deal with label changes?
    labels = {}
    fn = os.path.join(rootdir, labelsfn)
    with open(fn) as fh:
        # this file has no header so the first item is 0
        for idx, line in enumerate(fh):
            # label,objectname - maybe do something else with objectname later
            label = line.strip().split(",")[0]
            labels[label] = idx
    return labels


@timed
def readimgs(imgfn, rootdir='.'):
    # load the images into a big dictionary: keys = image name, values = index
    # TODO: How will we deal with label changes?
    images = {}
    fn = os.path.join(rootdir, imgfn)
    with open(fn) as fh:
        for idx, line in enumerate(fh):
            if idx == 0:
                continue # ignore the header
            # imgname is the only field we care about
            imgname = line.strip().split(",")[0]
            images[imgname] = idx - 1  # line 1 in the file is the header
    return images


# fields we care about for now are:
# ImageID (position 0)
# LabelName (position 2)
# Confidence (position 3)

@timed
def generate_counts(filename, labels, images, use_confidence=False, rootdir='.'):
    # filename: File containing the mapping of images to labels
    # labels: Dictionary of label names to label index, 0 indexed
    # images: Dictionary of image names to image index, 0 indexed
    fh = open(os.path.join(rootdir, filename), 'r')
    # convert to dictionary of keys scipy sparse matrix
    # store as dictionary with keys = tuples of image, label
    # values = count
    # this will be easier to convert to sparse matrix format later
    imglabels = {}
    for idx, line in enumerate(fh):
        # ignore line 0 which is the header
        if not idx:
            continue
        vals = line.split(",")
        imgname, _, labelname, confidence = vals[:4]
        labelidx = labels[labelname]
        imgidx = images[imgname]
        imglabels.setdefault((imgidx, labelidx), 0)
        score = 1    # if it was found in the file at all
        if use_confidence:
            confidence = float(confidence)
            score *= confidence
        imglabels[imgidx, labelidx] += score
    fh.close()
    return imglabels


def co_occurrence(imglabels):
    # imglabels: Dictionary of (image index, label index) to score
    sorted_keys = sorted(imglabels.keys())
    data = np.fromiter((imglabels[k] for k in sorted_keys),
                       dtype=float, count=len(sorted_keys))
    skarr = np.array(sorted_keys).T
    imglabels_sparse = scipy.sparse.coo_matrix((data, (skarr[0], skarr[1])))
    i = torch.LongTensor(skarr)
    v = torch.FloatTensor(data)
    imglabels_torch = torch.sparse.FloatTensor(i, v)
    return imglabels_torch, imglabels_sparse



if __name__ == '__main__':
    #rootdir = "/home/petra/data"
    rootdir = '.'

    # Download from https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv
    # The file is large, so not included in source control
    #fn = os.path.join(rootdir, "oidv6-train-annotations-human-imagelabels.csv")
    fn = "oidv6-train-annotations-bbox.csv"

    # Download from https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv
    labelsfn = 'oidv6-class-descriptions.csv'

    # Download from https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv
    imgfn = 'oidv6-train-images-with-labels-with-rotation.csv'


    labels = readlabels(labelsfn, rootdir=rootdir)
    images = readimgs(imgfn, rootdir=rootdir)
    imglabels = generate_counts(fn, labels, images, rootdir=rootdir)
    coo_pt, coo_scipy = co_occurrence(imglabels)

