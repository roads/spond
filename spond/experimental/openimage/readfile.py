import time
import subprocess
import itertools
import scipy.sparse

# Download from https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-human-imagelabels.csv
# The file is large, so not included in source control
#fn = "oidv6-train-annotations-human-imagelabels.csv"
fn = "oidv6-train-annotations-bbox.csv"

# Download from https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv
labelsfn = 'oidv6-class-descriptions.csv'

# Download from https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv
imgfn = 'oidv6-train-images-with-labels-with-rotation.csv'

# load the labels into a big dictionary: keys = labels, values = index
# TODO: How will we deal with label changes?

labels = {}
start = time.time()
with open(labelsfn) as fh:
    # this file has no header so the first item is 0
    for idx, line in enumerate(fh.readlines()):
        # label,objectname - maybe do something else with objectname later
        label = line.strip().split(",")[0]
        labels[label] = idx
end = time.time()
total = end-start

print(f"Done processing labels in {total} s")

images = {}
start = time.time()
with open(imgfn) as fh:
    for idx, line in enumerate(fh.readlines()):
        if idx == 0:
            continue # ignore the header
        # imgname is the only field we care about
        imgname = line.strip().split(",")[0]
        images[imgname] = idx - 1  # line 1 in the file is the header
end = time.time()
total = end-start

print(f"Done processing images in {total} s")


out = subprocess.run(["wc", "-l", fn], capture_output=True)

lines = int(out.stdout.decode().split(" ")[0])

# fields we care about for now are:
# ImageID (position 0)
# LabelName (position 2)
# Confidence (position 3)


def process(args):
    # args: tuple of (filename, start line number, end line number)
    # necessary for multiprocessing later.
    filename, start, end = args
    fh = open(filename, 'r')
    # convert to dictionary of keys scipy sparse matrix
    # Keys: image index
    # Values: set of label indices
    imglabels = {} #scipy.sparse.dok_matrix((Nlabels, Nimages))
    output = {}
    for line in itertools.islice(fh, start, end):
        vals = line.split(",")
        imgname, _, labelname, confidence = vals[:4]
        labelidx = labels[labelname]
        imgidx = images[imgname]
        confidence = float(confidence)
        imglabels.setdefault(imgidx, set())
        imglabels[imgidx].add(labelidx)
        output.setdefault(labelidx, {})
        # set what other labels occur with this one
        # probably want to do something with confidence as well
        # not store it unless it's above a certain threshold
        exists = imglabels[imgidx]
        for item in exists:
            if item != labelidx:   # don't set for self
                output[labelidx].setdefault(item, 0)
                # at this point, instead of just adding 1,
                # we could do other stuff with the pair
                output[labelidx][item] += 1
    fh.close()
    return imglabels, output


start = time.time()
# ignore the header, which is in line 1
imglabels, output = process((fn, 1, lines))
end = time.time()
total = end-start
print(f"Done processing associations in {total} s")
