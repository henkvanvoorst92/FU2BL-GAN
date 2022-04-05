from __future__ import division
import torch, random, sys#, cv2
from scipy.ndimage.interpolation import rotate,zoom
from scipy.ndimage import gaussian_filter
import numpy as np
import SimpleITK as sitk
from skimage.exposure import equalize_hist
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.interpolation import rotate
import scipy
from scipy.ndimage import binary_erosion
from utils.Utils import *
from skimage.transform import radon, iradon
from skimage import exposure
import seaborn as sns
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import torchvision
import torch
#functions and classes to augment images

# check: https://github.com/MIC-DKFZ/nnUNet


def clip_arr(arr, min=-100,max=500):
    return np.clip(arr,min,max)

def clip_image(img, min = -1024, max = 1900):
    '''
    clip img between HU values
    :param img: sitk image to be clipped
    :param min: minimum clip value
    :param max: maximum clip value
    :return:
    '''
    clampFilter = sitk.ClampImageFilter()
    if max!=None:
        clampFilter.SetUpperBound(max)
    if min!=None:
        clampFilter.SetLowerBound(min)
    return clampFilter.Execute(img)

class CLIP(object):
    """
    Clips the image intensities to min max values
    """
    def __init__(self,min,max,select_apply=[]):
        self.min = min
        self.max = max
        self.select_apply = select_apply

    def __call__(self, imgs_in):
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        imgs_out = []
        for i,img in enumerate(imgs_in):
            if self.select_apply[i]:
                if isinstance(img, np.ndarray):
                    img = clip_arr(img, self.min, self.max)
                elif isinstance(img, sitk.Image):
                    img = clip_image(img, self.min, self.max)
                else:
                    print('Type to clip not specified:\n',type(img))
            imgs_out.append(img)
        return imgs_out

class ErodeBrain(object):
    """
    Clips the image intensities to min max values
    """
    def __init__(self,n_erosion=2,background=-1024, return_mask=False, select_apply=[]):
        self.n_erosion = n_erosion
        self.background = background
        self.return_mask = return_mask
        self.select_apply = select_apply

    def __call__(self, imgs_in):

        # select apply can overrule the flag for applying the brain erosion
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        imgs_out, masks_out = [], []
        for i,img in enumerate(imgs_in):
            if self.select_apply[i]:
                if isinstance(img, np.ndarray):
                    mask = binary_erosion((img>self.background+100)*1, iterations=self.n_erosion)
                    img[mask==0]=img.min()

                elif isinstance(img, sitk.Image):
                    mask = sitk.BinaryThreshold(img, self.background+100, self.background+5000)

                    erode = sitk.BinaryErodeImageFilter()
                    erode.SetBackgroundValue(0)
                    erode.SetForegroundValue(1)
                    erode.SetKernelRadius((self.n_erosion,self.n_erosion))

                    mask = erode.Execute(mask)
                    img = sitk_apply_mask(mask,img, foreground=1, background=self.background)

                else:
                    print('Type to clip not specified:\n',type(img))
                masks_out.append(mask)

            imgs_out.append(img)
            

        if self.return_mask:
            return imgs_out, masks_out
        else:
            return imgs_out


def normalize_image(img, min = -1, max = 1 ):
    '''
    normalize the sitk image in range
    :param img: sitk image
    :param min: rescale minimum
    :param max: rescale maximum
    :return:
    '''
    normalizationFilter = sitk.RescaleIntensityImageFilter()
    normalizationFilter.SetOutputMaximum(max)
    normalizationFilter.SetOutputMinimum(min)
    return normalizationFilter.Execute(img)

def normalize_arr(arr, min=-1, max=1):
    # normalize the array between min and max values
    # of the input array to min-max given
    if arr.min()<0:
        arr = arr + arr.min()*-1
    arr_std = (arr - arr.min()) / (arr.max() - arr.min())
    arr_scaled = arr_std * (max - min) + min
    return arr_scaled

def normalize_arr_between(arr,low=-100,hi=800, min=-1, max=1, clip=False):
    # normalize array with input low-hi values to min-max range
    if clip:
        arr = np.clip(arr,low,hi)

    if low<0:
        arr = arr + low*-1
        hi += low*-1
        low = 0
        
    arr_std = (arr - low) / (hi - low)
    arr_scaled = arr_std * (max - min) + min
    return arr_scaled

class Normalize(object):
    """
    Normalizes the image intensities between min max
    """
    def __init__(self,input_lowhi=None,min=-1,max=1, type='float32', select_apply=[]):
        self.min = min
        self.max = max
        self.type = type
        self.input_lowhi = input_lowhi
        self.select_apply = select_apply

    def __call__(self, imgs_in):
        # select apply can overrule the flag for applying the normalization
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        imgs_out = []
        for i,img in enumerate(imgs_in):
            if self.select_apply[i]:
                if isinstance(img, np.ndarray):
                    if self.input_lowhi==None:
                        # normalize between the input array min and max values
                        img = normalize_arr(img, self.min, self.max)
                    elif len(self.input_lowhi)==2:
                        low, hi = self.input_lowhi
                        img = normalize_arr_between(img, low, hi, self.min, self.max, clip=True)
                    else:
                        print('Error input_lowhi is not properly initialized:', type(self.input_lowhi), self.input_lowhi)

                elif isinstance(img, sitk.Image):
                    img = normalize_image(img, self.min, self.max)
                else:
                    print('Type to normalize is wrong:\n',type(img))

            imgs_out.append(img.astype(self.type))

        return imgs_out


### code for 3D tensors
class RandomFlip(object):
    """
    Randomly horizontally flips the 
    given np array with a probability of 0.5
    """
    def __init__(self,p=1, ax=1, not_same=False, select_apply=[]): # ax dimensions (z,y,x) depends on input array
        self.p = p
        self.ax = ax
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob of flipping)
        self.select_apply = select_apply

    def __call__(self, imgs_in):

        # select apply can overrule the flag for applying the noise adding
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        imgs_out= []
        flag = random.random() < self.p
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                img = np.flip(img, axis=self.ax)

            imgs_out.append(img)

            if self.not_same:
                flag = random.random() < self.p

        return imgs_out

class RandomRotation(object):
    """
    Rotate 2 PIL images by a 
    random sample (same) angle
    """
    def __init__(self, angles, p=1, axes=(1,0), reshape=True, mode='nearest', not_same=False, select_apply=[]):
        """
        angles is tuple or int that
        p is the probability the rotation is performed
        represents the lower-upper bound
        of the random drawn angle to rotate
        axes to rotate are: (0,1,2):(z,y,x)
        (2,1):(x,y): axial rotat
        (2,0):(x,z): rotat to the side
        (1,0):(y,z): rotate to front/back
        """
        if isinstance(angles,int):
            # create lower and upper bound if integer is passed
            self.angles = (angles*-1,angles) 
        else:
            self.angles = angles # angles has to be like (min_rotate, max rotate); can be negative

        self.p = p
        self.axes = axes
        self.reshape = reshape
        self.mode = mode
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob and angle of rotation)
        self.select_apply = select_apply

    def __call__(self, imgs_in):

        # select apply can overrule the flag for applying the rotation
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        angle = np.random.randint(*self.angles)
        flag = random.random() < self.p
        imgs_out= []
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                img = rotate(img,axes=self.axes, 
                            angle=angle, 
                            reshape=self.reshape, 
                            mode=self.mode)

            imgs_out.append(img)
            if self.not_same:
                angle = np.random.randint(*self.angles)
                flag = random.random() < self.p
                
        return imgs_out

def unsharp_noise_mask(image, radius=1, amount=1):
    smoothed = gaussian_filter(image, radius)
    unsharp = image + amount * (image - smoothed)
    return unsharp

class UnsharpNoise(object):
    """
    Increases the noise present in an image:
    A gaussian filter with a defined radius
    is used to create a smoothed image to add with a
    certain factor (amount) multiplied to the
    original image. 
    Amount can also be a tuple to sample from.
    This augmentation occurs with 
    probability p. If same is used the same noise
    factor is used
    """
    def __init__(self,p,amount,radius=1,not_same=False, select_apply=[]):
        self.p = p
        self.radius = radius
        self.amount = amount
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob and degree of noise adding)
        self.select_apply = select_apply
        
    def __call__(self, imgs_in):

        # select apply can overrule the flag for applying the noise adding
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        # decide on the factor of noise adding
        if isinstance(self.amount, float) or isinstance(self.amount, int):
            factor = self.amount
        elif isinstance(self.amount,tuple):
            factor = round(random.uniform(*self.amount),1)
        else:
            print('Error amount is of wrong type:', type(self.amount))

        flag = random.random() < self.p
        imgs_out= []
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                img = unsharp_noise_mask(img, radius=1, amount=factor)

            imgs_out.append(img)

            if self.not_same:
                flag = random.random() < self.p
                if isinstance(self.amount,tuple):
                    factor = round(random.uniform(*self.amount),1)

        return imgs_out


class Smooth(object):
    """
    Smooths the images with a gaussian filter
    """
    def __init__(self,p,radius=1,repeats=1, not_same=False, select_apply=[]):
        self.p = p
        self.radius = radius
        self.repeats = repeats
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob of smoothing)
        self.select_apply = select_apply # can overrule the flag for augmentation (if false no smoothing)

    def __call__(self, imgs_in):
        
        # select apply can overrule the flag for applying the smoothing
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]


        flag = random.random() < self.p
        imgs_out= []
        for i, img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                for j in range(self.repeats):
                    img = gaussian_filter(img, self.radius)
            imgs_out.append(img)

            if self.not_same:
                flag = random.random() < self.p

        return imgs_out


class Zoom(object):
    """
    Zooms in or out with a factor for 
    both images based on:
    amount: a fixed float input or a 
    tuple to sample from
    """
    def __init__(self,p,amount, mode='nearest', not_same=False, select_apply=[]):
        self.p = p
        self.amount = amount
        self.mode = mode
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob of zooming)
        self.select_apply = select_apply

    def __call__(self, imgs_in):

        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        # decide on the factor of zooming used
        if isinstance(self.amount, float):
            factor  = self.amount
        elif isinstance(self.amount, tuple):
            # draw a random value from the tuple range
            factor = round(random.uniform(*self.amount),2)
        else:
            print('Error: amount input type is not compatible', 
                    type(self.amount))

        flag = random.random() < self.p
        imgs_out = []
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                img = zoom(img, factor, mode = self.mode)

            imgs_out.append(img)
            if self.not_same:
                flag = random.random() < self.p
                if isinstance(self.amount, tuple):
                    factor = round(random.uniform(*self.amount),2)
        
        return imgs_out


def crop_dim(img, dim, max_dim_shape = 512):
    """
    Crop an image in predefined dimension (dim) 
    with a maximum shape (max_dim_shape)
    """
    dim_shape = img.shape[dim]
    difference = dim_shape - max_dim_shape
    if difference>0:
        crop = int(np.floor(difference/2))
        if dim==0:
            img = img[crop:-(difference-crop),:]
        elif dim==1:
            img = img[:,crop:-(difference-crop)]
        elif dim==2:
            img = img[:,:,crop:-(difference-crop)]
    else:
        crop = None

    return img, crop

def crop_img(img, default_dims=(512,512)):
    """
    Implements crop_dim for multiple dimensions consecutively
    """
    crop_dims = []
    for dim,max_shape in enumerate(default_dims):
        img, crop = crop_dim(img, dim, max_shape)
        crop_dims.append(crop)
    return img, crop_dims

def paste_in_default_size(img, default_dims=(512,512)):
    """
    If the image is smaller than default_dims
    paste the image in an background 
    valued empty (emp) array
    """
    # initialize empty arr with min-value of img
    max0,max1 = default_dims
    emp = np.full((default_dims), img.min())
    # compute differences
    dt0, dt1 = (max0-img.shape[0]), (max1-img.shape[1])
    #if dt0<0 or dt1<0:
     #   print('Error: image is bigger than shape {}, crop first'
      #        .format(default_dims))
    
    # find start of dimension in empty arr
    if dt0>0:
        s0 = int(np.floor(dt0/2))
        low0 = s0
        hi0 = s0+img.shape[0]
    else:
        low0 = 0
        hi0 = img.shape[0]
    
    if dt1>0:
        s1 = int(np.floor(dt1/2))
        low1 = s1
        hi1 = s1+img.shape[1]
    else:
        low1 = 0
        hi1 = img.shape[1]        
        
    emp[low0:hi0, low1:hi1] = img

    return emp,[low0,hi0, low1,hi1]
  
class Resize_default(object):
    """
    Resizes the list of images to (512,512)
    """
    def __init__(self, default_dims=(512,512)): # ax dimensions (z,y,x) depends on input array
        # return dims is used during validation inference to reconstruct original shaped image
        self.default_dims = default_dims

    def __call__(self, imgs_in):
        imgs_out = []
        for img in imgs_in:
            if img.shape != self.default_dims:
                img, __ = crop_img(img,self.default_dims)
                img, __ = paste_in_default_size(img,self.default_dims)

            imgs_out.append(img)
                
        return imgs_out

def slicewise_HU(input_m,aggr=['mean', 'median'],clip_range=(-50,200)):
    """
    returns per slice of ctp series the mean for each timeframe
    
    input_m    -- input matrix of shape (time,z,x,y) --> obtain from func: get_ctp_matrix in bilateral_filter.py
    aggr       -- np aggregation operations for an array (outputs a single value)
    clip_range -- lower part is excluded, range is used for clipping
    """
    if clip_range!=None:
        lower, upper = clip_range
    aggr_funcs = [getattr(np,a) for a in aggr] # get numpy funcs
    cols = [a+'_HU' for a in aggr]
    
    out = []
    # iterate over slices (!!! write these operations as matrix operation !!!!)
    for slix in range(0,input_m.shape[1]): 
        # iterate over timeframes
        for t in range(input_m.shape[0]): # i is time
            data = input_m[t,slix,:,:].flatten()
            if clip_range!=None:
                data = data[(data>=lower)]
                data = clip_arr(data,lower,upper)
            if len(data)<1:
                continue
                
            aggr_dta = [f(data) for f in aggr_funcs]
            out.append([t,slix,*aggr_dta])

    return pd.DataFrame(out,columns=['timeframe', 'slix',*cols])

def radon_beamnoise(img,theta,clip_sinogram=(0,'mean'), filter_name='ramp', visualize=False):
    """
    img     --- an np array of a 2D image (most often 512x512)
    theta   --- set of angles for sinogram reconstruction
    clip_sinogram --- range to clip sinogram values (also np functions allowed)
    filter_name   --- filter method used for invers radon transform (see skimage docs)
    
    this function computes the sinogram of an image (img)
    with the radon transform (Filtered Back Projection)
    subsequently the sinogram values are clipped and an 
    inverse radon transform is made
    
    the inverse radon transform is beam hardening noise that
    can be added to the input image (img)
    
    """
    sinogram = radon(clip_arr(img,0,10000), theta=theta)
    
    min_s, max_s = clip_sinogram
    if not (isinstance(max_s, int) or isinstance(max_s, float)):
        max_s = getattr(np,max_s)(sinogram[sinogram>0].flatten())
    if visualize:
        plt.imshow(sinogram)
        plt.show()
        sns.distplot(sinogram.flatten())
        plt.title(str(max_s))
        plt.show()
    sinogram = clip_arr(sinogram, min=min_s,max=max_s)
    return iradon(sinogram,filter=filter_name) #the inverse radon transform image

class RadonBeamNoise(object):
    """
    Uses radon and inverse radon tranform to generate 
    beam hardening artefacts.

    """
    def __init__(self,p,amount, max_shape = 512, 
                 background=-1024, clip_img = (0,250),
                 clip_sino=(0,'mean'), filter_name='ramp', 
                 not_same=False, select_apply=[]):
        self.p = p # probability of noise adding
        self.amount = amount # amount of noise (multiplication of noise) to add
        self.theta = np.linspace(0., 180., max_shape, endpoint=False) # radon transform rotation
        self.background = background # used to filter out noise generated in background
        self.clip_sino = clip_sino # range used to clip sinogram reconstructed from radon transform (see radon_beamnoise docs)
        self.clip_img = clip_img # the input image is clipped before radon transform (otherwise to dispersed values)
        self.filter_name = filter_name # filter to use for iradon transform
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob of zooming)
        self.select_apply = select_apply # selection of input images list to perform augmentation for

    def __call__(self, imgs_in):

        # decide on which of th einput scans to perform noise augmentation
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        # decide on the factor of noise to add
        if isinstance(self.amount, float) or isinstance(self.amount, int):
            factor = self.amount
        elif isinstance(self.amount,tuple):
            factor = round(random.uniform(*self.amount),1)
        else:
            print('Error amount is of wrong type:', type(self.amount))


        flag = random.random() < self.p
        imgs_out= []
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                #compute mask to keep out background beam hardening
                # mask = img>self.background
                # #compute median HU inside foreground (for HU adjustment)
                # pre_median = np.median(img[mask].flatten())

                # noise = radon_beamnoise(clip_arr(img.copy(),self.clip_img[0],self.clip_img[1]),
                #                         theta=self.theta,
                #                         clip_sinogram=self.clip_sino, 
                #                         filter_name=self.filter_name, 
                #                         visualize=False)

                # img = (img+noise*factor*(mask*1)) #only add noise inside the mask
                # post_median = np.median(img[mask].flatten())
                # # Adjust median HU values
                # delta = (np.zeros_like(img)+post_median-pre_median)*(mask*1)
                # img-= delta
                noise = radon_beamnoise(clip_arr(img.copy(),self.clip_img[0],self.clip_img[1]),
                                        theta=self.theta,
                                        clip_sinogram=self.clip_sino, 
                                        filter_name=self.filter_name, 
                                        visualize=False)


                img = self.add_RBN(self,img,noise,factor)

            imgs_out.append(img.astype('int16'))

            if self.not_same: # resample probability and factor per iteration
                flag = random.random() < self.p
                if isinstance(self.amount,tuple):
                    factor = round(random.uniform(*self.amount),1)

        return imgs_out

    def add_RBN(self,img,noise,factor):
        #adds RBN to image and adjust shift in median HU

        mask = ((img>self.background)&(img<self.clip_img[1]))
        pre_median = np.median(img[mask].flatten()) 

        img = (img+noise*factor*(mask*1)) #only add noise inside the mask
        post_median = np.median(img[mask].flatten())
        # Adjust median HU values
        delta = (np.zeros_like(img)+post_median-pre_median)*(mask*1)
        img-= delta
        return img

class MultiHUChannel(object):
    """
    Creates different HU clipping and normalization ranges images stacked channelwise
    ranges       --- list of tuples with low,hi normalization
    channel_dim  --- Dimension to stack the new images across
    output_range --- upper and lower normalization boundary default = (-1,1)
    
    """
    def __init__(self,ranges,channel_dim=-1,output_range=(-1,1), select_apply=[]):
        self.ranges = ranges
        self.min_out, self.max_out = output_range
        self.select_apply = select_apply
        self.channel_dim = channel_dim
        
    def __call__(self, imgs_in):
        # volume is a D,H,W np array
    
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]
        
        imgs_out, masks_out = [], []
        for i,img in enumerate(imgs_in):
            if self.select_apply[i]:
                tmp_imgs = []
                for low,hi in self.ranges:
                    tmp_imgs.append(normalize_arr_between(img.copy(),
                                                        low=low,hi=hi, 
                                                        min=self.min_out, max=self.max_out, clip=True))
                imgs_out.append(np.stack(tmp_imgs,axis=self.channel_dim))
            else: # scale only the first channel
                low,hi = self.ranges[0]
                imgs_out.append(normalize_arr_between(img.copy(),
                                        low=low,hi=hi, 
                                        min=self.min_out, max=self.max_out, clip=True))
            
        return imgs_out


def multi_channel_img(img,ranges=[(-100,250),(250,1000)],minmax=(-1,1), channeldim=0):
    """
    converts single channel image (greyscale) to multichannel image
    by stacking different intensity ranges clipped and normalized 
    onto eachother
    """
    out = []
    for (low,hi) in ranges:
        out.append(normalize_arr_between(img.copy(),low=low,hi=hi, min=minmax[0], max=minmax[1], clip=True))
    return np.stack(out, axis=channeldim)
    
class MCI(object):
    # class version of multi_channel_image
    # used to stack multiple HU ranges
    def __init__(self,
             ranges=[(-100,250),(250,1000)], 
             minmax=(-1,1), 
             channeldim=0):
        
        self.ranges = ranges
        self.minmax = minmax
        self.channeldim = channeldim
    
    def __call__(self, img):
        return multi_channel_img(img,self.ranges, self.minmax, self.channeldim)
    
 
class FU2BL_PP(object):
    """
    Specific FU2BL preprocessing of images
    
    """
    def __init__(self,types=['norm_bl', 'stack_ranges', 'mask'], 
                 ranges=[(-100,250),(250,1000)], 
                 minmax=(-1,1), 
                 channeldim=0, 
                 MatchHE=None): 
        
        self.types = types
        self.ranges = ranges
        self.minmax = minmax
        self.channeldim = channeldim
        self.MatchHE = MatchHE # matches histograms to template or eachother --> requires both BL and FU images not to be normalized
        # MatchHe is either None, ('template',Augment.Class, prob), or ('eachother',0,prob)
        
    def __call__(self, imgs_in):
        imgs_out= []
        if isinstance(self.MatchHE,tuple):
            flag = random.random() < self.MatchHE[2]
            #print(flag)

        for t,img in zip(self.types,imgs_in):
            if t=='norm_bl':
                if not isinstance(self.MatchHE,tuple):
                    # this is default img processing setting
                    img = multi_channel_img(img.copy(),ranges=self.ranges[0:1],minmax=self.minmax, channeldim=self.channeldim)
                elif self.MatchHE[0]=='template': # match BL image to pre-defined template (do the same for fu stack)
                    # use a probability to decide if HE adjustment occurs
                    if flag:
                        img = self.MatchHE[1](img)
                    img = multi_channel_img(img,ranges=self.ranges[0:1],minmax=self.minmax, channeldim=self.channeldim)
                elif self.MatchHE[0]=='eachother': # match BL image to histogram of FU
                    # use a probability to decide if HE adjustment occurs
                    if flag:
                        refimg = clip_arr(imgs_in[self.types.index('stack_ranges')].copy(),self.ranges[0][0],self.ranges[0][1])  # the FU scan
                        img = match_histograms(clip_arr(img.copy(),self.ranges[0][0],self.ranges[0][1]), refimg, self.ranges[0], multichannel=False)
                        img = multi_channel_img(img,ranges=self.ranges[0:1],minmax=self.minmax, channeldim=self.channeldim)
                else:
                    print('Error')

            elif t=='stack_ranges':
                if not isinstance(self.MatchHE,tuple):
                    # this is default img processing setting
                    img = multi_channel_img(img,ranges=[(0,200),(200,1000)])
                elif self.MatchHE[0]=='template': # match BL image to pre-defined template (do the same for fu stack)
                    # use a probability to decide if HE adjustment occurs
                    if flag:
                        img = self.MatchHE[1](img)
                    img = multi_channel_img(img,ranges=self.ranges,minmax=self.minmax, channeldim=self.channeldim)
                elif self.MatchHE[0]=='eachother': # match BL image to histogram of FU
                    # use a probability to decide if HE adjustment occurs
                    #refimg = clip_arr(imgs_in[self.types.index('stack_ranges')].copy(),self.ranges[0],self.ranges[1])  # the FU scan
                    #img = match_histograms(clip_arr(img,self.ranges[0],self.ranges[1]), refimg, self.ranges, multichannel=False)
                    img = multi_channel_img(img,ranges=self.ranges,minmax=self.minmax, channeldim=self.channeldim)
                else:
                    print('Error')

            elif t=='stack_ranges_HE':
                tmp = multi_channel_img(img.copy(),ranges=self.ranges,minmax=self.minmax)
                img = clip_arr(img.copy(),self.ranges[0][0],self.ranges[0][1]) 
                HE = normalize_arr(image_histogram_equalization(img.copy()),self.minmax[0],self.minmax[1])
                img = np.concatenate([tmp,np.expand_dims(HE,axis=0)], axis=0)
            elif t=='mask':
                img = img

            imgs_out.append(img.astype(np.float32))

        return imgs_out

class RescaleDefault(object):
    """
    Rescales the list of images to default size
    """
    def __init__(self, default_dim=256, select_antialiasing=[]): # ax dimensions (z,y,x) depends on input array
        # return dims is used during validation inference to reconstruct original shaped image
        self.default_dims = (default_dim,default_dim)
        self.select_antialiasing = select_antialiasing
        
    def __call__(self, imgs_in):
        if self.select_antialiasing is None or len(self.select_antialiasing)!=len(imgs_in):
            self.select_antialiasing= [True for i in imgs_in]
            
        imgs_out = []
        for ix,img in enumerate(imgs_in):
            if img.shape != self.default_dims:
                maxax = np.max(img.shape)
                frac = self.default_dims[0]/maxax
                img = rescale(img, frac, anti_aliasing=self.select_antialiasing[ix])

            if img.shape[0]!=img.shape[1]: # adjust if dims are not equal
                img, __ = crop_img(img,self.default_dims)
                img, __ = paste_in_default_size(img,self.default_dims)

            imgs_out.append(img)
                
        return imgs_out

class GammaScaling(object):
    """
    Performs gamma scaling of input images intensity
    best defaults: gamma_range=(0.9,1.1) or (0.95,1.05), gain_range=1
    """
    def __init__(self,p, gamma_range=(0.9,1.1), gain_range=1,background=-100, not_same=False, select_apply=[]):
        self.p = p
        self.gamma_range = gamma_range
        self.gain_range = gain_range
        self.background = background
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob of zooming)
        self.select_apply = select_apply

    def __call__(self, imgs_in):

        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        # decide on the factor of gamma scaling
        if isinstance(self.gamma_range, float) or isinstance(self.gamma_range, int):
            gamma = self.gamma_range
        elif isinstance(self.gamma_range, tuple):
            # draw a random value from the tuple range
            gamma = round(random.uniform(*self.gamma_range),3)
        else:
            print('Error: amount input type is not compatible', 
                    type(self.gamma_range))

        #decide on the gain to use
        if isinstance(self.gain_range, float) or isinstance(self.gain_range, int):
            gain = self.gain_range
        elif isinstance(self.gain_range, tuple):
            # draw a random value from the tuple range
            gain = round(random.uniform(*self.gain_range),3)
        else:
            print('Error: amount input type is not compatible', 
                    type(self.gain_range))

        flag = random.random() < self.p
        imgs_out = []
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                m = img>self.background
                im = exposure.adjust_gamma(clip_arr(img.copy(),0,10000),gamma=gamma,gain=gain)
                img[m] = im[m]

            imgs_out.append(img)
            if self.not_same:
                flag = random.random() < self.p
                if isinstance(self.gamma_range, tuple):
                    # draw a random value from the tuple range
                    gamma = round(random.uniform(*self.gamma_range),3)
                if isinstance(self.gain_range, tuple):
                    # draw a random value from the tuple range
                    gain = round(random.uniform(*self.gain_range),3)

        return imgs_out

class LogScaling(object):
    """
    Performs Log scaling of input images intensity
    best defaults: gamma_range=(8,12)
    """
    def __init__(self,p, log_range=(8,12),background=-100, not_same=False, select_apply=[]):
        self.p = p
        self.log_range = log_range
        self.background = background
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob of zooming)
        self.select_apply = select_apply

    def __call__(self, imgs_in):

        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        # decide on the factor of gamma scaling
        if isinstance(self.log_range, float) or isinstance(self.log_range, int):
            log = self.log_range
        elif isinstance(self.log_range, tuple):
            # draw a random value from the tuple range
            log = round(random.uniform(*self.log_range),1)
        else:
            print('Error: amount input type is not compatible', 
                    type(self.log_range))
            
        flag = random.random() < self.p
        imgs_out = []
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                m = img>self.background
                im = exposure.adjust_log(clip_arr(img.copy(),0,10000),gain=log)
                img[m] = im[m]

            imgs_out.append(img)
            
            if self.not_same:
                flag = random.random() < self.p
                if isinstance(self.log_range, tuple):
                    # draw a random value from the tuple range
                    log = round(random.uniform(*self.log_range),1)
        return imgs_out

class AddIntensity(object):
    """
    Adds intensity offset to images
    """
    def __init__(self,p, intensity_range=(-20,20),background=-100, not_same=False, select_apply=[]):
        self.p = p
        self.intensity_range = intensity_range
        self.background = background
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob of zooming)
        self.select_apply = select_apply

    def __call__(self, imgs_in):
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        # decide on the factor of gamma scaling
        if isinstance(self.intensity_range, float) or isinstance(self.intensity_range, int):
            add_intensity = self.intensity_range
        elif isinstance(self.intensity_range, tuple):
            # draw a random value from the tuple range
            add_intensity = np.random.randint(*self.intensity_range)
        else:
            print('Error: amount input type is not compatible', 
                    type(self.intensity_range))
            
        flag = random.random() < self.p
        imgs_out = []
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                img[img>self.background] += add_intensity

            imgs_out.append(img)
            
            if self.not_same:
                flag = random.random() < self.p
                if isinstance(self.intensity_range, tuple):
                    # draw a random value from the tuple range
                    add_intensity = round(random.uniform(*self.intensity_range),1)
        
        return imgs_out

class Brightness(object):
    """
    Adjusts brightness in image (multiplies all pixels with factor)
    default values brightness_range between 0.8 and 1.2
    """
    def __init__(self,p, brightness_range=(.8,1.2),background=-100, not_same=False, select_apply=[]):
        self.p = p
        self.brightness_range = brightness_range
        self.background = background
        self.not_same = not_same # if true, unpaired augmentation (each image has own prob of zooming)
        self.select_apply = select_apply

    def __call__(self, imgs_in):
        if self.select_apply is None or len(self.select_apply)!=len(imgs_in):
            self.select_apply = [True for i in imgs_in]

        # decide on the factor of gamma scaling
        if isinstance(self.brightness_range, float) or isinstance(self.brightness_range, int):
            brightness = self.brightness_range
        elif isinstance(self.brightness_range, tuple):
            # draw a random value from the tuple range
            brightness = round(random.uniform(*self.brightness_range),3)
        else:
            print('Error: amount input type is not compatible', 
                    type(self.brightness_range))
            
        flag = random.random() < self.p
        imgs_out = []
        for i,img in enumerate(imgs_in):
            if flag and self.select_apply[i]:
                img[img>self.background] *= brightness

            imgs_out.append(img)
            
            if self.not_same:
                flag = random.random() < self.p
                if isinstance(self.brightness_range, tuple):
                    # draw a random value from the tuple range
                    brightness = round(random.uniform(*self.brightness_range),3)
        
        return imgs_out

class SelectAug(object):
    """"
    mode='one': With probability p performs any of one augmentations
    mode='any': With probability p per method performs augmentation (could be joint)
    
    Receives a list of defined classes with augmentation
    """
    def __init__(self,p,Augs,mode='one'):
        self.p = p
        self.Augs = Augs
        self.mode=mode
    
    def __call__(self,imgs_in, arg2=None):
        
        flag = random.random() < self.p
        if self.mode=='one': # chosse one augmentation method
            Aug = [random.choice(self.Augs)]
            #print(Aug)s
        elif self.mode=='any': # pass all with prob predefined in Augs classes
            Aug = self.Augs

        imgs_out = imgs_in
        for A in Aug:
            if flag:
                if arg2 is None:
                    imgs_out = A(imgs_out)
                else:
                    imgs_out = A(imgs_out,arg2)
            
        return imgs_out


