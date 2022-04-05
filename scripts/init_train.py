
from utils.Samplers import *
from utils.Augment import *
from utils.Loader_RBN import Compose, HDF5Dataset
import numpy as np
import torch

def init_augment(opt):
    p_augment = opt.p_augment

    horflip = RandomFlip(p=p_augment, ax=1,select_apply=[True,True])
    RRxy = RandomRotation(angles=20,p=p_augment, axes=(1,0), reshape=True,select_apply=[True,True])
    Z = Zoom(p_augment,(.8,1.2),select_apply=[True,True])
    # only add noise to NCCT --> distort ground truth
    UN = UnsharpNoise(opt.p_contrast_augment,(.5,1.2), not_same=True, select_apply=[True, False]) # for 2D apply on NCCT only
    S = Smooth(opt.p_contrast_augment, not_same=False, select_apply=[True, False]) # for 2D
    RD = Resize_default(default_dims=(512,512))

    AI = AddIntensity(p=1, intensity_range=(-20,20),background=-1000, not_same=False, select_apply=[True, False])
    B = Brightness(p=1, brightness_range=(.8,1.2),background=-1000, not_same=False, select_apply=[True, False])
    # Non-linear contrast adjustments
    GS = GammaScaling(p=1, gamma_range=(0.9,1.1), gain_range=1,background=-1000, not_same=False, select_apply=[True, False])
    LS = LogScaling(p=1, log_range=(9,11),background=-1000, not_same=False, select_apply=[True, False])
    #selects one of the above intensity augmentations (with a probability)
    SelectContr = SelectAug(p=opt.p_contrast_augment,Augs=[AI,B,GS],mode='one')

    Clip = CLIP(min=-1024, max=3000) # skull has to large variations >300 so clip more narrow
    RD = Resize_default(default_dims=(512,512))
    

    special_PP = FU2BL_PP(types=['norm_bl', 'stack_ranges'], 
                          ranges=opt.HU_ranges, minmax=(-1,1), 
                          channeldim=1, MatchHE=None)#('template',HM2T,.1))#MatchHE can be set to None, now 10% gets matched to template HU range
    #('template',HM2T, .1)
    train_transforms = Compose([horflip,RRxy,Z,UN,S,Clip,RD,SelectContr,special_PP])

    return train_transforms

def init_loader(opt):

    train_transforms = init_augment(opt)

    TSS = SliceSamplerTwoStaged(paired=True, margin1=(.1,.95), margin2 = (.2,.8),prob_fact = 3,input_n_slices = 1)

    train_dataset = HDF5Dataset(opt.train_filepath,
                data_cache_size=opt.cache_size, 
                slice_sampler = TSS, 
                transform = train_transforms,
                prob_RBN = opt.p_RBN, # probability of adding beam hardening noise
                factor_RBN = opt.factor_RBN, # factor of RBN added
                mode='train', 
                return_brainmask=False)

    train_ldr = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=opt.batch_size, 
                    num_workers=opt.n_workers,
                    shuffle=True, 
                    drop_last=False)
    return train_ldr, train_dataset, TSS

