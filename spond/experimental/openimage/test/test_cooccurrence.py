import unittest

from spond.experimental.openimage import readfile

class TestProcessing(unittest.TestCase):

    def setUp(self):
        self.rootdir = "."
        # image metadata file, contains image labels
        self.imgfn = "test-image-labels.csv"
        # labels metadata file, contains labels to descriptions
        self.labelsfn = "test-labels.csv"
        # annotations file, contains what labels map to what images
        self.datafn = "test-annotations.csv"

        self.imgdict = readfile.readimgs(self.imgfn, self.rootdir)
        self.labelsdict = readfile.readimgs(self.labelsfn, self.rootdir)

    def lookup(self, image, label):
        # given the label strings, return the indexes in the dictionaries
        return (
            self.imgdict[image],
            self.labelsdict[label]
        )

    def test_process_images(self):
        imgdict = self.imgdict
        # there are only 3 images in this file
        self.assertEquals(len(imgdict), 3)
        self.assertEquals(min(imgdict.values()), 0)
        self.assertEquals(max(imgdict.values()), len(imgdict) - 1)

    def test_process_labels(self):
        labelsdict = self.labelsdict
        # there are only 9 images in this file
        self.assertEquals(len(labelsdict), 9)
        self.assertEquals(min(labelsdict.values()), 0)
        self.assertEquals(max(labelsdict.values()), len(labelsdict) - 1)

    def test_cooccurrence_matrix_use_confidence(self):
        imgdict = readfile.readimgs(self.imgfn, self.rootdir)
        labelsdict = readfile.readimgs(self.labelsfn, self.rootdir)
        imglabels, coo = readfile.generate_cooccurrence(
            self.datafn, labelsdict, imgdict, rootdir=self.rootdir,
            use_confidence=True
        )
        # 064jy_j and 05r655 co-occur twice:
        # once each in images 497919baa5f92e69 and 0899cae1f10e5f9f
        i, j = self.labelsdict["/m/064jy_j"], self.labelsdict["/m/05r655"]
        self.assertEquals(coo[(i, j)], coo[(j, i)])
        self.assertEquals(coo[(i, j)], 2)
        # 064kdv_ and 0271t do not co-occur
        i, j = self.labelsdict["/m/064kdv_"], self.labelsdict["/m/0271t"]
        self.assertRaises(KeyError, lambda: coo[(i, j)])
        # We requested to use_confidence, and
        # 0643t and 02smb6 occur in 0899cae1f10e5f9f but with confidence 0
        # so they should present but with co-occurrence score of 0,
        # with every other label in 0899cae1f10e5f9f
        zeroconf = ('/m/0643t', '/m/02smb6')
        # all these other items are present only once in 0899cae1f10e5f9f
        present = ['/m/0271t',
                   '/m/0118n_9r',
                   '/m/04dr76w',
                   '/m/020p1v']
        for label in present:
            for other in present:
                if label == other:
                    continue
                # each pair should have a score of 1.
                i, j = self.labelsdict[label], self.labelsdict[other]
                self.assertEquals(coo[(i, j)], coo[(j, i)])
                self.assertEquals(coo[(i, j)], 1)
            for other in zeroconf:
                i, j = self.labelsdict[label], self.labelsdict[other]
                self.assertEquals(coo[(i, j)], coo[(j, i)])
                self.assertEquals(coo[(i, j)], 0)

    def test_cooccurrence_matrix_without_confidence(self):
        imgdict = readfile.readimgs(self.imgfn, self.rootdir)
        labelsdict = readfile.readimgs(self.labelsfn, self.rootdir)
        imglabels, coo = readfile.generate_cooccurrence(
            self.datafn, labelsdict, imgdict, rootdir=self.rootdir,
            use_confidence=False
        )
        # 064jy_j and 05r655 co-occur twice:
        # once each in images 497919baa5f92e69 and 0899cae1f10e5f9f
        i, j = self.labelsdict["/m/064jy_j"], self.labelsdict["/m/05r655"]
        self.assertEquals(coo[(i, j)], coo[(j, i)])
        self.assertEquals(coo[(i, j)], 2)
        # 064kdv_ and 0271t do not co-occur
        i, j = self.labelsdict["/m/064kdv_"], self.labelsdict["/m/0271t"]
        self.assertRaises(KeyError, lambda: coo[(i, j)])
        present = ['/m/0271t',
                   '/m/0118n_9r',
                   '/m/04dr76w',
                   '/m/0643t',
                   '/m/02smb6',
                   '/m/020p1v']
        for label in present:
            for other in present:
                if label == other:
                    continue
                # each pair should have a score of 1.
                i, j = self.labelsdict[label], self.labelsdict[other]
                self.assertEquals(coo[(i, j)], coo[(j, i)])
                self.assertEquals(coo[(i, j)], 1)
