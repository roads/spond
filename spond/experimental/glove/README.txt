The data all currently lives in /home/petra/data. 

There are separate directories for openimages and audioset.

The Python environment used to run all experiments is:

/home/petra/envs/pytorch1938

They were usually run from an IPython interactive shell:

/home/petra/envs/pytorch1938/bin/ipython

and then

%run whateverscript.py

The main important libraries and their versions are:

* Python 3.8 - older than 3.8 may be possible, it is just untested. 
* PyTorch 1.9 - this is a hard requirement, as something didn't work with 1.7 
* torch-two-sample from the branch https://github.com/rekcahpassyla/torch-two-sample/tree/pytorch-fix
* nltk 

I tried to make a pull request for torch-two-sample, but I could not
get the unit tests working and I won't do a PR without being able to run those. 

The main scripts that generate embeddings are as follows.
Each script has comments which should be read
as they give an idea of which classes do what. 

- glove_layer.py which runs deterministic GloVe embeddings
- probabilistic_glove.py which runs probabilistic GloVe embeddings
- aligned_glove.py which runs aligned GloVe embeddings of
  OpenImages and AudioSet together.

Results are saved in the directory

/home/petra/spond/spond/experimental/glove/results

Unfortunately this is a bit disorganised and this path is used hard coded
in various files instead of being a config parameter. 

The main scripts for analysis are as follows. They should be thought of
as only templates or examples. They don't contain any generation of
results. 

- analyse.py which calculates similarity matrices, correlations, entropies etc
  for probabilistic single-domain GloVe embeddings. This script only runs
  one domain at a time and must be run twice.
- analyse_aligned.py which calculates the same metrics for aligned GloVe
  embeddings (so both domains at once).
- inspect_results.py which contains various functions to munge data into
  the right structures for plotting. This file is not so useful for analysis
  as it contains mostly code for generating my images, but it is left here
  as an example.

There are also 3 scripts for similarity analysis which all do pretty much
the same thing - comparing the embedding pair similarity with various
human similarity judgement datasets.
- hsj_similarity.py: Compare with MTURK-771
- ilsvrc_similarity.py: Compare with enhanced ILSVRC dataset
- wordnet_similarity.py: Compare with WordNet using LCH similarity score.

Finally there is samples_similarity.py which takes N (in this case, 100)
samples of the embeddings and calculates similarity scores based on the
mean of those samples. 
