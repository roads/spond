import os
import numpy as np
import tempfile
import shutil
import torch
from unittest import TestCase

from spond.experimental.glove.glove_layer import GloveSimple


class GloveLayerTest(TestCase):

    def test_save_load(self):
        train_embeddings_file = '/home/petra/data/audioset/glove_audioset.pt'
        train_data = torch.load(train_embeddings_file, map_location=torch.device('cpu'))
        nemb, dim = train_data['wi.weight'].shape
        model = GloveSimple(batch_size=100,
            train_embeddings_file=train_embeddings_file,
            nconcepts=nemb, dim=dim,
            train_cooccurrence_file='/home/petra/data/audioset/co_occurrence.pt',
                            double=True)
        # make somewhere to keep it
        tmpdir = tempfile.mkdtemp()
        outfile = os.path.join(tmpdir, 'glove.pt')
        try:
            torch.save(model, outfile)
            rt = torch.load(outfile)
            for attrname in ('wi', 'bi', 'wj', 'bj'):
                assert torch.allclose(
                    getattr(model.glove_layer, attrname).weight,
                    getattr(rt.glove_layer, attrname).weight
                ), f'Roundtrip of {attrname} was not equal'
            # can't compare the sparse co_occurrence because torch blows up
            # using allclose on the sparse structure.
            assert torch.allclose(
                model.glove_layer.coo_dense,
                rt.glove_layer.coo_dense
            ), 'Roundtrip of coo_dense was not equal'
            for attrname in ('x_max', 'alpha'):
                assert np.allclose(
                    getattr(model.glove_layer, attrname),
                    getattr(rt.glove_layer, attrname)
                ), f'Roundtrip of {attrname} was not equal'
        finally:
            # always delete this
            shutil.rmtree(tmpdir)

