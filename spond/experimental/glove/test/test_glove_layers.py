# This file contains save/load tests for all the glove layers
# GloveLayer, ProbabilisticGloveLayer, AlignedGloveLayer
import os
import numpy as np
import tempfile
import shutil
import torch
from unittest import TestCase

from spond.experimental.glove.glove_layer import GloveSimple
from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove


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
                ), f'Roundtrip of GloveLayer.{attrname} was not equal'
            # can't compare the sparse co_occurrence because torch blows up
            # using allclose on the sparse structure.
            assert torch.allclose(
                model.glove_layer.coo_dense,
                rt.glove_layer.coo_dense
            ), 'Roundtrip of GloveLayer.coo_dense was not equal'
            for attrname in ('x_max', 'alpha'):
                assert np.allclose(
                    getattr(model.glove_layer, attrname),
                    getattr(rt.glove_layer, attrname)
                ), f'Roundtrip of GloveLayer.{attrname} was not equal'
        finally:
            # always delete this
            shutil.rmtree(tmpdir)


class ProbabilisticGloveLayerTest(TestCase):

    def test_save_load(self):
        train_embeddings_file = '/home/petra/data/audioset/glove_audioset.pt'
        train_cooccurrence_file = '/home/petra/data/audioset/co_occurrence.pt'
        model = ProbabilisticGlove(
            train_embeddings_file,
            use_pretrained=False,
            batch_size=500,
            seed=1,
            train_cooccurrence_file=train_cooccurrence_file)
        # make somewhere to keep it
        tmpdir = tempfile.mkdtemp()
        outfile = os.path.join(tmpdir, 'glove.pt')
        try:
            torch.save(model, outfile)
            rt = torch.load(outfile)

            # check model attributes
            for attrname in ('seed', 'train_embeddings_file',
                             'train_cooccurrence_file', 'use_pretrained',
                             'batch_size'):
                self.assertEquals(
                    getattr(model, attrname),
                    getattr(rt, attrname)
                ), f'Roundtrip of ProbabilisticGlove.{attrname} was not equal'

            # check layer attributes
            for attrname in ('wi_mu', 'bi_mu', 'wi_rho', 'bi_rho'):
                assert torch.allclose(
                    getattr(model.glove_layer, attrname).weight,
                    getattr(rt.glove_layer, attrname).weight
                ), f'Roundtrip of ProbabilisticGloveLayer.{attrname} was not equal'
            # can't compare the sparse co_occurrence because torch blows up
            # using allclose on the sparse structure.
            assert torch.allclose(
                model.glove_layer.coo_dense,
                rt.glove_layer.coo_dense
            ), 'Roundtrip of ProbabilisticGloveLayer.coo_dense was not equal'
            for attrname in ('x_max', 'alpha'):
                assert np.allclose(
                    getattr(model.glove_layer, attrname),
                    getattr(rt.glove_layer, attrname)
                ), f'Roundtrip of ProbabilisticGloveLayer.{attrname} was not equal'
        finally:
            # always delete this
            shutil.rmtree(tmpdir)

