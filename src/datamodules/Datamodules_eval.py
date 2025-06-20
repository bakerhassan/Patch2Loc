import torchio as tio
from lightning import LightningDataModule
from typing import Optional
import pandas as pd
import torchio
import src.datamodules.create_dataset as create_dataset

class Brats21(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(Brats21, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_test = cfg.path.Brats21.IDs.test
        self.csv = {}
        states = ['test']
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'Brats21'

            self.csv[state]['img_path'] = cfg.path.pathBase + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase  + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase  + self.csv[state]['seg_path']

            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1',cfg.mode).str.replace('FLAIR.nii.gz',f'{cfg.mode.lower()}.nii.gz')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                # self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else :
                # self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def test_dataloader(self):
        return tio.SubjectsLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=False)




class ATLAS_v2(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(ATLAS_v2, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}

        self.csvpath_val = cfg.path.ATLAS_v2.IDs.val
        self.csvpath_test = cfg.path.ATLAS_v2.IDs.test

        self.csv = {}
        states = ['test']

        # self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'ATLAS_v2'

            self.csv[state]['img_path'] = cfg.path.pathBase  + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase  + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase  + self.csv[state]['seg_path']

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                # self.val_eval = create_dataset.Eval(self.csv['val'][0:8], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:8], self.cfg)
            else :
                # self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def test_dataloader(self):
        return  tio.SubjectsLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=False)


class MSLUB(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(MSLUB, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.MSLUB.IDs.val
        self.csvpath_test = cfg.path.MSLUB.IDs.test
        self.csv = {}
        states = ['test']

        # self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'MSLUB'

            self.csv[state]['img_path'] = cfg.path.pathBase + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase  + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase  + self.csv[state]['seg_path']
            
            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('MSLUB/t1',f'MSLUB/{cfg.mode}').str.replace('t1.nii.gz',f'{cfg.mode}.nii.gz')
    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                # self.val_eval = create_dataset.Eval(self.csv['val'][0:4], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:4], self.cfg)
            else :
                # self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)

    def test_dataloader(self):
        return  tio.SubjectsLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=False)


class WMH(LightningDataModule):

    def __init__(self, cfg, fold= None):
        super(WMH, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        self.imgpath = {}
        self.csvpath_val = cfg.path.WMH.IDs.val
        self.csvpath_test = cfg.path.WMH.IDs.test
        self.csv = {}
        states = ['test']

        # self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'WMH'

            self.csv[state]['img_path'] = cfg.path.pathBase  + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase  + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = cfg.path.pathBase  + self.csv[state]['seg_path']

            if cfg.mode != 't1':
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1',f'{cfg.mode}')
                
    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'val_eval'):
            if self.cfg.sample_set: # for debugging
                # self.val_eval = create_dataset.Eval(self.csv['val'][0:4], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:4], self.cfg)
            else :
                # self.val_eval = create_dataset.Eval(self.csv['val'], self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'], self.cfg)
    def test_dataloader(self):
        return  tio.SubjectsLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=False)
