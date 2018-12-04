from __future__ import absolute_import

import unittest
import os
import random

from got10k.datasets import GOT10k, OTB, VOT, DTB70, TColor128, ImageNetVID


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.data_dir = 'data'

    def tearDown(self):
        pass

    def test_got10k(self):
        root_dir = os.path.join(self.data_dir, 'GOT-10k')
        # without meta
        dataset = GOT10k(root_dir, subset='train')
        self._check_dataset(dataset)
        # with meta
        dataset = GOT10k(root_dir, subset='val', return_meta=True)
        self._check_dataset(dataset)

    def test_otb(self):
        root_dir = os.path.join(self.data_dir, 'OTB')
        dataset = OTB(root_dir)
        self._check_dataset(dataset)
    
    def test_vot(self):
        root_dir = os.path.join(self.data_dir, 'vot2017')
        # without meta
        dataset = VOT(root_dir, anno_type='rect')
        self._check_dataset(dataset)
        # with meta
        dataset = VOT(root_dir, anno_type='rect', return_meta=True)
        self._check_dataset(dataset)

    def test_dtb70(self):
        root_dir = os.path.join(self.data_dir, 'DTB70')
        dataset = DTB70(root_dir)
        self._check_dataset(dataset)
    
    def test_tcolor128(self):
        root_dir = os.path.join(self.data_dir, 'Temple-color-128')
        dataset = TColor128(root_dir)
        self._check_dataset(dataset)

    def test_vid(self):
        root_dir = os.path.join(self.data_dir, 'ILSVRC')
        dataset = ImageNetVID(root_dir, subset=('train', 'val'))
        self._check_dataset(dataset)
    
    def _check_dataset(self, dataset):
        n = len(dataset)
        self.assertGreater(n, 0)
        inds = random.sample(range(n), min(n, 100))
        for i in inds:
            img_files, anno = dataset[i][:2]
            self.assertEqual(len(img_files), len(anno))


if __name__ == '__main__':
    unittest.main()
