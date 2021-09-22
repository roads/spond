# This file contains save/load tests for all the glove layers
# GloveLayer, ProbabilisticGloveLayer, AlignedGloveLayer
import os
import numpy as np
import tempfile
import shutil
import torch
from unittest import TestCase
import pytorch_lightning as pl

from spond.experimental.glove.glove_layer import GloveSimple
from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove
from spond.experimental.glove.aligned_glove import AlignedGlove, DataDictionary


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
        trainer = pl.Trainer(gpus=0, max_epochs=1, progress_bar_refresh_rate=20)
        trainer.fit(model)
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
        trainer = pl.Trainer(gpus=0, max_epochs=1, progress_bar_refresh_rate=20)
        trainer.fit(model)
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


class AlignedGloveLayerTest(TestCase):

    def test_save_load(self):
        x_cooc_file = y_cooc_file = '/home/petra/data/audioset/co_occurrence.pt'
        x_labels_file = y_labels_file = '/home/petra/data/audioset/class_labels.csv'
        all_labels_file = x_labels_file
        batch_size = 1000
        x_dim = y_dim = 6
        seed = 1
        probabilistic = True
        supervised = True
        mmd = 100
        save = True
        epochs = 100
        datadict = DataDictionary(
            x_cooc=torch.load(x_cooc_file),
            x_labels_file=x_labels_file,
            y_cooc=torch.load(y_cooc_file),
            y_labels_file=y_labels_file,
            all_labels_file=all_labels_file,
            intersection_plus=None
        )

        model = AlignedGlove(batch_size,
                             data=datadict,
                             x_embedding_dim=x_dim,  # dimension of x
                             y_embedding_dim=y_dim,  # dimension of y
                             seed=seed,
                             probabilistic=probabilistic,
                             supervised=supervised, mmd=mmd,
                             save_flag=save,
                             max_epochs=epochs)
        trainer = pl.Trainer(gpus=0, max_epochs=1, progress_bar_refresh_rate=20)
        trainer.fit(model)
        # make somewhere to keep it
        tmpdir = tempfile.mkdtemp()
        outfile = os.path.join(tmpdir, 'glove.pt')
        try:
            torch.save(model, outfile)
            rt = torch.load(outfile)

            # check model attributes
            for attrname in (
                    'seed', 'batch_size', 'x_embedding_dim', 'y_embedding_dim',
                    'probabilistic', 'supervised', 'max_epochs', 'save_flag'
                ):
                self.assertEquals(
                    getattr(model, attrname),
                    getattr(rt, attrname)
                ), f'Roundtrip of AlignedGlove.{attrname} was not equal'

            for emb_name in ('x_emb', 'y_emb'):
                layer = getattr(model.aligner, emb_name)
                rt_layer = getattr(rt.aligner, emb_name)
                # check layer attributes
                for attrname in ('wi_mu', 'bi_mu', 'wi_rho', 'bi_rho'):
                    assert torch.allclose(
                        getattr(layer, attrname).weight,
                        getattr(rt_layer, attrname).weight
                    ), f'Roundtrip of AlignedGlove.aligner.{emb_name}.{attrname} was not equal'
                # can't compare the sparse co_occurrence because torch blows up
                # using allclose on the sparse structure.
                assert torch.allclose(
                    layer.coo_dense,
                    rt_layer.coo_dense
                ), f'Roundtrip of AlignedGlove.aligner.{emb_name}.coo_dense was not equal'
                for attrname in ('x_max', 'alpha'):
                    assert np.allclose(
                        getattr(layer, attrname),
                        getattr(rt_layer, attrname)
                    ), f'Roundtrip of AlignedGlove.aligner.{emb_name}.{attrname} was not equal'
        finally:
            # always delete this
            shutil.rmtree(tmpdir)
