import torch
from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule
import torchio
import src.datamodules.create_dataset as create_dataset
from typing import Optional
import pandas as pd


class IXI(LightningDataModule):

    def __init__(self, cfg, fold = None):
        super(IXI, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        # IXI

        self.cfg.permute = False # no permutation for IXI
        self.imgpath = {}
        self.csvpath_train = cfg.path.IXI.IDs.train[fold]
        self.csvpath_val = cfg.path.IXI.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI.IDs.test
        self.csv = {}
        states = ['train','val','test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        if cfg.mode == 't2':
            keep_t2 = pd.read_csv(cfg.path.IXI.keep_t2) # only keep t2 images that have a t1 counterpart

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI'


            self.csv[state]['img_path'] = cfg.path.pathBase + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase  + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

            if cfg.mode == 't2': 
                self.csv[state] = self.csv[state][self.csv[state].img_name.isin(keep_t2['0'].str.replace('t2','t1'))]
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1','t2')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'train'):
            if self.cfg.sample_set: # for debugging
                self.train = create_dataset.Train(self.csv['train'][0:50],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'][0:50],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'][0:4],self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'][0:4],self.cfg)
            else: 
                self.train = create_dataset.Train(self.csv['train'],self.cfg) 
                self.val = create_dataset.Train(self.csv['val'],self.cfg)
                self.val_eval = create_dataset.Eval(self.csv['val'],self.cfg)
                self.test_eval = create_dataset.Eval(self.csv['test'],self.cfg)
    
    def train_dataloader(self):
        return torchio.SubjectsLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=True, drop_last=self.cfg.get('droplast',False))

    def val_dataloader(self):
        return torchio.SubjectsLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=False)

    def val_eval_dataloader(self):
        return torchio.SubjectsLoader(self.val_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=False)

    def test_eval_dataloader(self):
        return torchio.SubjectsLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=False)


class IXIPatches(LightningDataModule):

    def __init__(self, cfg, fold = None):
        super(IXIPatches, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get('preload',True)
        # load data paths and indices
        # IXI
        self.patch_size = int(self.cfg.imageDim[0] * self.cfg.patch_percentage)
        self.cfg.permute = False # no permutation for IXI


        self.imgpath = {}
        self.csvpath_train = cfg.path.IXI.IDs.train[fold]
        self.csvpath_val = cfg.path.IXI.IDs.val[fold]
        self.csvpath_test = cfg.path.IXI.IDs.test
        self.csv = {}
        states = ['train','val','test']

        self.csv['train'] = pd.read_csv(self.csvpath_train)
        self.csv['val'] = pd.read_csv(self.csvpath_val)
        self.csv['test'] = pd.read_csv(self.csvpath_test)
        if cfg.mode == 't2':
            keep_t2 = pd.read_csv(cfg.path.IXI.keep_t2) # only keep t2 images that have a t1 counterpart

        for state in states:
            self.csv[state]['settype'] = state
            self.csv[state]['setname'] = 'IXI'


            self.csv[state]['img_path'] = cfg.path.pathBase + self.csv[state]['img_path']
            self.csv[state]['mask_path'] = cfg.path.pathBase  + self.csv[state]['mask_path']
            self.csv[state]['seg_path'] = None

            if cfg.mode == 't2': 
                self.csv[state] = self.csv[state][self.csv[state].img_name.isin(keep_t2['0'].str.replace('t2','t1'))]
                self.csv[state]['img_path'] = self.csv[state]['img_path'].str.replace('t1','t2')

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        if not hasattr(self,'train'):
            if self.cfg.sample_set: # for debugging
                self.train = create_dataset.Train(self.csv['train'][0:4],self.cfg, True, True, self.patch_size) 
                self.val = create_dataset.Train(self.csv['val'][0:4],self.cfg,True,  True, self.patch_size) 
                self.val_eval = create_dataset.Train(self.csv['val'][0:4],self.cfg,True,  True, self.patch_size) 
                self.test_eval = create_dataset.Train(self.csv['test'][0:4],self.cfg,True,  True, self.patch_size) 
            else: 
                self.train = create_dataset.Train(self.csv['train'],self.cfg,True, True, self.patch_size) 
                self.val = create_dataset.Train(self.csv['val'],self.cfg,True, True, self.patch_size) 
                self.val_eval = create_dataset.Train(self.csv['val'],self.cfg,True, True, self.patch_size) 
                self.test_eval = create_dataset.Train(self.csv['test'],self.cfg,True, True, self.patch_size) 
    
    def contrastive_collate(self, batch):
        anchors = torch.stack([item['anchor'] for item in batch])
        positives = torch.stack([item['positive'] for item in batch])
        return {
            'anchors': anchors,
            'positives': positives,
        }

    def train_dataloader(self):
        return torchio.SubjectsLoader(self.train, batch_size=self.cfg.batch_size, 
                                      num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, 
                                      shuffle=True, drop_last=self.cfg.get('droplast',False),
                                      collate_fn=self.contrastive_collate)

    def val_dataloader(self):
        return torchio.SubjectsLoader(self.val, batch_size=self.cfg.batch_size, 
                                      num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory,
                                        shuffle=False,
                                        collate_fn=self.contrastive_collate)

    def val_eval_dataloader(self):
        return torchio.SubjectsLoader(self.val_eval, batch_size=1,
                                       num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, 
                                       shuffle=False)

    def test_eval_dataloader(self):
        return torchio.SubjectsLoader(self.test_eval, batch_size=1, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, shuffle=False)
