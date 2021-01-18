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
    """
    Parameters
    ----------
    filename: str
        File containing the labels
    rootdir: str, optional, default to "."
        If passed, will be the directory the files live in

    Returns
    -------
    labels: {str: int}
        Dictionary of labels to label index, 0 indexed
    """
    labels = {}
    fn = os.path.join(rootdir, labelsfn)
    with open(fn) as fh:
        # this file has no header so the first item is 0
        for idx, line in enumerate(fh):
            # label,objectname - maybe do something else with objectname later
            # for now we only care about label
            label = line.strip().split(",")[0]
            labels[label] = idx
    return labels


@timed
def readimgs(imgfn, rootdir='.'):
    """
    Parameters
    ----------
    filename: str
        File containing the images
    rootdir: str, optional, default to "."
        If passed, will be the directory the files live in

    Returns
    -------
    images: {str: int}
        Dictionary of image names to image index, 0 indexed
    """
    images = {}
    fn = os.path.join(rootdir, imgfn)
    with open(fn) as fh:
        for idx, line in enumerate(fh):
            if idx == 0:
                continue # ignore the header
            # imgname is the first field, and the only field we care about
            imgname = line.strip().split(",")[0]
            images[imgname] = idx - 1  # line 1 in the file is the header
    return images


@timed
def generate_cooccurrence(filename, labels, images, use_confidence=False, rootdir='.'):
    """
    Parameters
    ----------
    filename: str
        File containing the mapping of images to labels
    labels: {str: int}
        Dictionary of label names to label index, 0 indexed
    images: {str: int}
        Dictionary of image names to image index, 0 indexed
    use_confidence: boolean, optional, default to False
        If set to True, the confidence value in the file will be used when
        calculating the co-occurrence score.
    rootdir: str, optional, default to "."
        If passed, will be the directory the files live in

    Returns
    -------
    imglabels: {str: {str: float}}
        First level keys are image index,
        second level keys are label index,
        second level values are score.
        Score = 1 if confidence is not used, else 1 * confidence
    coo: {(int, int): float}
        Co-occurrence dictionary.
        Keys are (image index, other image index)
        Values are sum of scores of the pair occurring over all image pairs.
        Reverse pairs (other image index, image index) are also stored
        so this matrix is symmetric.
    """
    fh = open(os.path.join(rootdir, filename), 'r')
    # keys: image index, values: {label index, score}
    imglabels = {}
    # keys: label index, values: {img_index: confidence}
    conf = {}
    # keys: [(label index, other label index)], values: score
    coo = {}
    for idx, line in enumerate(fh):
        # ignore line 0 which is the header
        if not idx:
            continue
        vals = line.split(",")
        # fields we care about for now are:
        # ImageID (position 0)
        # LabelName (position 2)
        # Confidence (position 3)
        imgname, _, labelname, confidence = vals[:4]
        labelidx = labels[labelname]
        imgidx = images[imgname]
        imglabels.setdefault(imgidx, {})
        conf.setdefault(labelidx, {})
        imglabels[imgidx].setdefault(labelidx, 0)
        conf[labelidx].setdefault(imgidx, 0)
        score = 1    # if it was found in the file at all
        if use_confidence:
            confidence = float(confidence)
            score *= confidence
            conf[labelidx][imgidx] = confidence
        imglabels[imgidx][labelidx] += score
        for otherlabel in imglabels[imgidx]:
            if labelidx == otherlabel:
                continue
            coo.setdefault((labelidx, otherlabel), 0)
            coo.setdefault((otherlabel, labelidx), 0)
            # if using confidence, we should multiply by the confidence
            # of the other label
            used_score = score
            if use_confidence:
                otherconf = conf[otherlabel][imgidx]
                used_score = score * otherconf
            coo[(labelidx, otherlabel)] += used_score
            coo[(otherlabel, labelidx)] += used_score
    fh.close()
    return imglabels, coo


def to_pytorch(coo):
    """
    Parameters
    ----------
    coo: {(int, int): float}
        Dictionary of (label index, other label index) to score
        that represents co-occurrence

    Returns
    -------
    coo_torch: torch.SparseTensor
        representing the co-occurrence matrix
    """
    sorted_keys = sorted(coo.keys())
    data = np.fromiter((coo[k] for k in sorted_keys),
                       dtype=float, count=len(sorted_keys))
    skarr = np.array(sorted_keys).T
    i = torch.LongTensor(skarr)
    v = torch.FloatTensor(data)
    coo_torch = torch.sparse.FloatTensor(i, v)
    return coo_torch


def to_scipy(sptensor):
    """
    Parameters
    ----------
    sptensor: torch.SparseTensor
        Co-occurrence matrix in sparse tensor form

    Returns
    -------
    coo_scipy: scipy.sparse.csr_matrix
        representing the co-occurrence matrix
    """
    # convert Pytorch tensor to scipy sparse matrix.
    # This will be the path to deserialisation- not going to save the
    # scipy sparse matrix to file, as it isn't supported.
    # Returns matrix in CSR format; this can be changed to any other format
    # easily using the scipy API.
    values = sptensor._values()
    indices = sptensor._indices()
    return scipy.sparse.csr_matrix(
        (values.cpu().numpy(), indices.cpu().numpy()), shape=sptensor.shape)


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
    imglabels, coo = generate_cooccurrence(fn, labels, images, rootdir=rootdir)
    coo_pt = to_pytorch(coo)

    torch.save(coo_pt, 'image_labels.pt')
