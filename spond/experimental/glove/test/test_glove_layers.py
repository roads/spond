# This file contains save/load tests for all the glove layers
# GloveLayer, ProbabilisticGloveLayer, AlignedGloveLayer
import os
import numpy as np
import tempfile
import traceback
import shutil
import torch
from unittest import TestCase
import pytorch_lightning as pl

from spond.experimental.glove.glove_layer import GloveSimple
from spond.experimental.glove.probabilistic_glove import ProbabilisticGlove
from spond.experimental.glove.aligned_glove import AlignedGlove, DataDictionary


class GloveLayerTest(TestCase):

    @classmethod
    def compare_glove_layers(cls, model_layer, rt_layer):
        attrnames = ['wi', 'bi']
        # AlignedGlove with probabilistic=False will have double set to False
        # normal GloveLayer will use double = True
        if model_layer.double:
            attrnames += ['wj', 'bj']

        for attrname in attrnames:
            assert torch.allclose(
                getattr(model_layer, attrname).weight,
                getattr(rt_layer, attrname).weight
            ), f'Roundtrip of GloveLayer.{attrname} was not equal'
        # can't compare the sparse co_occurrence because torch blows up
        # using allclose on the sparse structure.
        assert torch.allclose(
            model_layer.coo_dense,
            rt_layer.coo_dense
        ), 'Roundtrip of GloveLayer.coo_dense was not equal'
        for attrname in ('x_max', 'alpha'):
            assert np.allclose(
                getattr(model_layer, attrname),
                getattr(rt_layer, attrname)
            ), f'Roundtrip of GloveLayer.{attrname} was not equal'

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
            self.compare_glove_layers(model.glove_layer, rt.glove_layer)
        finally:
            # always delete this
            shutil.rmtree(tmpdir)


class ProbabilisticGloveLayerTest(TestCase):

    @classmethod
    # extract into separate function so that AlignedGloveLayerTest can use it
    def compare_glove_layers(cls, model_layer, rt_layer):
        # check layer attributes
        for attrname in ('wi_mu', 'bi_mu', 'wi_rho', 'bi_rho'):
            assert torch.allclose(
                getattr(model_layer, attrname).weight,
                getattr(rt_layer, attrname).weight
            ), f'Roundtrip of ProbabilisticGloveLayer.{attrname} was not equal'
        # can't compare the sparse co_occurrence because torch blows up
        # using allclose on the sparse structure.
        assert torch.allclose(
            model_layer.coo_dense,
            rt_layer.coo_dense
        ), 'Roundtrip of ProbabilisticGloveLayer.coo_dense was not equal'
        for attrname in ('x_max', 'alpha'):
            assert np.allclose(
                getattr(model_layer, attrname),
                getattr(rt_layer, attrname)
            ), f'Roundtrip of ProbabilisticGloveLayer.{attrname} was not equal'

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
                np.testing.assert_equal(
                    getattr(model, attrname),
                    getattr(rt, attrname)
                ), f'Roundtrip of ProbabilisticGlove.{attrname} was not equal'

            self.compare_glove_layers(model.glove_layer, rt.glove_layer)
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
        supervised = True
        mmd = 100
        save = False
        epochs = 100
        datadict = DataDictionary(
            x_cooc=torch.load(x_cooc_file),
            x_labels_file=x_labels_file,
            y_cooc=torch.load(y_cooc_file),
            y_labels_file=y_labels_file,
            all_labels_file=all_labels_file,
            intersection_plus=None
        )

        # test both cases of probabilistic.
        # If probabilistic=True then the glove_layer will be ProbabilisticGloveLayer
        # and GloveLayer if not.
        for probabilistic, test_cls in [(True, ProbabilisticGloveLayerTest),
                                        (False, GloveLayerTest)]:

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

                # check attributes for each of the aligner layers
                for emb_name in ('x_emb', 'y_emb'):
                    layer = getattr(model.aligner, emb_name)
                    rt_layer = getattr(rt.aligner, emb_name)
                    try:
                        test_cls.compare_glove_layers(
                            layer, rt_layer
                        )
                    except:
                        tb = traceback.format_exc()
                        raise AssertionError(
                            f"probabilistic={probabilistic}: "
                            f"AlignedGlove.{emb_name}: {tb}"
                        )
            finally:
                # always delete this
                shutil.rmtree(tmpdir)
