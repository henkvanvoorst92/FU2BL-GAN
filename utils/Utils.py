import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os, math,torch
from numba import jit, njit, prange
from torch import nn
import SimpleITK as sitk
# skull strip scripts
from scipy.ndimage.measurements import label

def store_df(input, sav_loc, tpe, cols=None, separate=None):

    if isinstance(input,list):
        df_out = pd.DataFrame(input).reset_index(drop=True)
    elif isinstance(input,pd.DataFrame):
        df_out = input
    else:
        print('input type not compatible',type(input))
    

    if isinstance(cols,list):
        df_out.columns = cols
    if separate is not None:
        sav_loc+='_'+str(separate)
        
    sav_loc += tpe
    if os.path.exists(sav_loc):
        if tpe=='.xlsx':
            tmp = pd.read_excel(sav_loc)
        elif tpe=='.pic':
            tmp = pd.read_pickle(sav_loc)
        elif tpe=='.ftr':
            tmp = pd.read_feather(sav_loc)
        else:
            print('Error wrong type of extension:', tpe)
        if tmp.shape != (0,0): #sometimes written tmp is empty
            if 'Unnamed: 0' in tmp.columns:
                tmp = tmp.drop(columns=['Unnamed: 0'])
            if isinstance(cols,list):
                tmp.columns = cols
            df_out = pd.concat([df_out,tmp], axis=0).reset_index(drop=True)

    if tpe=='.xlsx':
        df_out.to_excel(sav_loc)
    elif tpe=='.pic':
        df_out.to_pickle(sav_loc)
    elif tpe=='.ftr':
        df_out.to_feather(sav_loc)


def np2itk(arr,original_img):
	img = sitk.GetImageFromArray(arr)
	img.SetSpacing(original_img.GetSpacing())
	img.SetOrigin(original_img.GetOrigin())
	img.SetDirection(original_img.GetDirection())
	# this does not allow cropping (such as removing thorax, neck)
	#img.CopyInformation(original_img) 
	return img

def sitk_dilate_mask(mask,radius_mm, dilate_2D=False):
	
	radius_3d = [int(math.floor(radius_mm / mask.GetSpacing()[0])),
			 int(math.floor(radius_mm / mask.GetSpacing()[1])),
			 int(math.floor(radius_mm / mask.GetSpacing()[2]))]
	if dilate_2D:
		radius_3d[2] = 0
	
	dilate = sitk.BinaryDilateImageFilter()
	dilate.SetBackgroundValue(0)
	dilate.SetForegroundValue(1)
	dilate.SetKernelRadius(radius_3d)
	return dilate.Execute(mask)

def RescaleInterceptHU(img):
	# if RescaleIntercept >-100 perform this transformation to get appropriate ranges
	img+=1024
	px_mode = 3096+1024
	img[img>=px_mode] = img[img>=px_mode] - px_mode
	img-=1024
	return img

def set_background(arr, mask, background=-1024):
	out = arr[mask==0] = background
	return out

def rtrn_np(img): # returns numpy array from torch tensor (on cuda)
	return img.detach().cpu().numpy()


def np_dice(y_true,y_pred,add=1):
	return (2*(y_true*y_pred).sum())/(y_true.sum()+y_pred.sum())




@njit(parallel=True,fastmath=True, nogil=True)
def remove_empty_slices(CT, minpixvals=-800, margin=1):
	"""
	Removes slices with a sum of pixel values <= maxpixval
	function uses numba njit for faster execution (speed x5)
	
	Input:
	CT: 3d np array with dimensions (x,y,z)
	maxpixval: maximum summed value of slice
	margin: selects n slices under/above the cut slices
	return_upper_lower: if the lower-upper bounds of the 
	original CT should be returned
	"""
	# lower indicates if lower bound of brain has been passed
	# upper indicates if the upper bound of brain has been passed
	lower, upper = False, False
	z_dim_CT = CT.shape[0]
	for i in range(0,z_dim_CT):
		sl = CT[i,:,:]
		pix_max = sl.max()
		# to select lower bound
		if (pix_max>minpixvals) & (lower==False):
			lower=True 
			lower_ix = i-margin
		# to select upper bound
		if (pix_max<=minpixvals) & (lower==True) & (upper==False):
			upper = True
			upper_ix = i+margin+1
			if upper_ix>z_dim_CT:
				upper_ix = z_dim_CT
				
	CT_out = CT[lower_ix:upper_ix,:,:]

	return CT_out, lower_ix, upper_ix

	
# function uses np not sitk
def apply_mask(m,img, foreground_m=1, background=-1024):
	if isinstance(m, sitk.SimpleITK.Image):
		m = sitk.GetArrayFromImage(m)
	if isinstance(img, sitk.SimpleITK.Image):
		img = sitk.GetArrayFromImage(img)
	img[m==foreground_m]=background
	return img

def sitk_apply_mask(m,img, foreground_m=0, background=None, sitk_type = sitk.sitkInt32):
	"""
	Applies mask (m) to an image, sets background of image
	returns and image with only mask foreground
	"""
	m = sitk.Cast(m,sitk_type)
	img = sitk.Cast(img,sitk_type)
	if foreground_m==0:
		mf = sitk.MaskNegatedImageFilter()
	elif foreground_m==1:
		mf = sitk.MaskImageFilter()
	if background!=None:
		mf.SetOutsideValue(background)
	return mf.Execute(img, m)

def sitk_dilate_mm(mask,kernel_mm, background=0, foreground=1):
	
	if isinstance(kernel_mm,int):
		k0 = k1 = k2 = kernel_mm
	elif isinstance(kernel_mm,tuple) or isinstance(kernel_mm,list):
		k0, k1, k2 = kernel_mm
	
	kernel_rad = (int(np.floor(k0/mask.GetSpacing()[0])),
			 int(np.floor(k1/mask.GetSpacing()[1])),
			 int(np.floor(k2/mask.GetSpacing()[2])))
	
	dilate = sitk.BinaryDilateImageFilter()
	dilate.SetBackgroundValue(background)
	dilate.SetForegroundValue(foreground)
	dilate.SetKernelRadius(kernel_rad)
	return dilate.Execute(mask)

def sitk_erode_mm(mask,kernel_mm, background=0, foreground=1):
	
	if isinstance(kernel_mm,int):
		k0 = k1 = k2 = kernel_mm
	elif isinstance(kernel_mm,tuple) or isinstance(kernel_mm,list):
		k0, k1, k2 = kernel_mm
	
	kernel_rad = (int(np.floor(k0/mask.GetSpacing()[0])),
			 int(np.floor(k1/mask.GetSpacing()[1])),
			 int(np.floor(k2/mask.GetSpacing()[2])))
	
	erode = sitk.BinaryErodeImageFilter()
	erode.SetBackgroundValue(background)
	erode.SetForegroundValue(foreground)
	erode.SetKernelRadius(kernel_rad)
	return erode.Execute(mask)

def dcm2niix(dcm2niix_exe, filename, output_dir, input_dir):
	#!activate root
	command = dcm2niix_exe + " -f "+filename+" -p y -z y -o"+ ' "'+ output_dir + '" "' + input_dir+ '"'
	os.system(command)

def assert_resliced_or_tilted(path,scanname='NCCT', ID=''):
	resl_tilted = [os.path.join(path,f) for f in os.listdir(path) \
				   if ('tilt' in f.lower() or 'eq' in f.lower()) and scanname.lower() in f.lower()]
	if len(resl_tilted)>0:
		p_ncct = resl_tilted [0]
		print(ID, scanname,'tilted or resliced:', '\n', p_ncct, '\n n adjusted:',len(resl_tilted))
		adjusted = True
	else:
		p_ncct = os.path.join(path,scanname+'.nii.gz')
		adjusted = False
	return p_ncct, adjusted

def sitk_flip_AP(img):
	return sitk.Flip(img, [False, True, False])

def compute_volume(mask):
	# mask is an sitk image
	# used to compute the volume in ml for
	sp = mask.GetSpacing()
	vol_per_vox = sp[0]*sp[1]*sp[2]
	
	m = sitk.GetArrayFromImage(mask)
	voxels = m.sum()
	#volume in ml
	tot_volume = vol_per_vox*voxels/1000
	return tot_volume

def largest_connected_component(img,min_threshold=-400, background=-1024):
	"""
	Retrieves largest connected component mask
	"""
	# compute connected components (in 3D)
	cc = sitk.ConnectedComponent(img>min_threshold)
	stats = sitk.LabelIntensityStatisticsImageFilter()
	stats.Execute(cc,img)
	max_size = 0
	# get largest connected component
	for l in stats.GetLabels():
		if stats.GetPhysicalSize(l)>max_size:
			max_label = l
			max_size = stats.GetPhysicalSize(l)
	# return mask
	return sitk.BinaryThreshold(cc, max_label, max_label+1e-2)

def remove_small_cc(seg,min_count=100):
	# filters out small connected components
	labels, nc = label(seg) # uses scipy.ndimage.measurements
	unique, counts = np.unique(labels, return_counts=True) # unique connected components
	v = unique[counts>min_count] # cc's larger than min_count
	out = ((labels*np.isin(labels,v)*1)*(seg>0)>0)*1 #combine all the above in a binary segmentation
	return out


def lower_upper_ix(mask,
				   z_difference=150,
				   z = -1,
				   foreground=1,
				   min_area=1000):

	"""
	Runs in a max from top to bottom of the head, finds
	the top slice where area>min_area and goes down 
	subsequently to z_difference mm below that point to
	return the bottom slice
	"""
	
	zdim = mask.GetSize()[z]
	for i in range(0,zdim):
		slice_id = zdim - i - 1
		slice_mask = mask[:, :, slice_id]
		label_info_filter = sitk.LabelStatisticsImageFilter()
		label_info_filter.Execute(slice_mask,slice_mask)
		area = label_info_filter.GetCount(foreground) * mask.GetSpacing()[0] * mask.GetSpacing()[1]
		if area>min_area:
			break
	top_slice = slice_id
	max_distance = mask.GetSpacing()[z]*zdim # the z-distance that is available in the mask
	bottom_slice = top_slice - int(z_difference/mask.GetSpacing()[z])
	if bottom_slice<0:
		bottom_slice = 0
			
	return (bottom_slice, top_slice, area, max_distance)


def MultipleMorphology(mask, 
						operations,
						mm_rads, 
						foreground=1):
	"""
	Consecutively performs multiple morphology operations
	"""
	if len(operations)!=len(mm_rads):
		print('Error: number of operations is not equal to number of radius (mm_rads)')
	
	rads_3d = []
	for rad in mm_rads:
		tmp = (int(math.floor(rad / mask.GetSpacing()[0])),
				 int(math.floor(rad / mask.GetSpacing()[1])),
				 int(math.floor(rad / mask.GetSpacing()[2])))
		rads_3d.append(tmp)
		
	for r,oper in zip(rads_3d, operations):
		oper.SetBackgroundValue(abs(foreground-1))
		oper.SetForegroundValue(foreground)
		oper.SetKernelRadius(r)
		mask = oper.Execute(mask)
			  
	return mask

def np_slicewise(mask, funcs, repeats=1):
	"""
	Applies a list of functions iteratively (repeats) slice by slice of an 3D np volume
	"""
	if isinstance(mask,sitk.SimpleITK.Image):
		mask = sitk.GetArrayFromImage(mask)

	out = np.zeros_like(mask)
	for sliceno in range(mask.shape[0]):
		m = mask[sliceno,:,:]
		for r in range(repeats):
			for func in funcs:
				m = func(m)
		out[sliceno,:,:] = m
	return out

def itk_erode_mask(mask, n_times=1, min_max_thres=(-1000,10000), no_z_erode=True):
	radius_3d = [int(math.floor(n_times / mask.GetSpacing()[0])),
			 int(math.floor(n_times / mask.GetSpacing()[1])),
			 int(math.floor(n_times / mask.GetSpacing()[2]))]

	if no_z_erode:
		radius_3d[2] = 0
   
	erode = sitk.BinaryErodeImageFilter()
	erode.SetBackgroundValue(0)
	erode.SetForegroundValue(1)
	erode.SetKernelRadius(radius_3d) 
	mask = erode.Execute(mask)
	return sitk.BinaryFillhole(mask)

def itk_erode_mask(mask, n_times=1, min_max_thres=(-1000,10000)):
	radius_3d = (int(math.floor(1/ mask.GetSpacing()[0])),
			 int(math.floor(1 / mask.GetSpacing()[1])),
			 int(math.floor(1 / mask.GetSpacing()[2])))
   
	erode = sitk.BinaryErodeImageFilter()
	erode.SetBackgroundValue(0)
	erode.SetForegroundValue(1)
	erode.SetKernelRadius(radius_3d) 
	for i in range(n_times):
		mask = erode.Execute(mask)
		mask = sitk.BinaryFillhole(mask)
	return mask

## this code comes from the CTA2NCCT train file and might interfere with similar named functions above
class Morphology(nn.Module):
	"""
	Pytorch implementation:
	Class performs erosion (iters<0) or dilation (iters>0) for a specific
	number of iterations on the foreground mask of an image. 
	The main benefit is that these operations can be performed on a 
	GPU during training if device is set to 'cuda'. This results
	in up to 20x (10x more likely) faster mask computation. On 'cpu' this operation is
	slower than multiple scipy erosion or dilation steps.
	"""
	
	def __init__(self, iters=35, background=-1, 
					connect=8, type=torch.float32, 
					device='cuda', combine='OR', dim3D=False):
		super(Morphology, self).__init__()

		self.dim3D = dim3D
		if iters>=0:
			self.operation = 'dilate'
			self.iters = iters
		else:
			self.operation = 'erode'
			self.iters = abs(iters)

		self.background = background
		self.type = type
		self.device = device
		self.connectivity = connect
		self.combine = combine # how to combine multiple images into a mask
		
		if connect==8:
			if not self.dim3D:
				self.kernel = torch.tensor([
						[1, 1, 1],
						[1, 1, 1],
						[1, 1, 1] ], 
						dtype=self.type
						).unsqueeze(0).unsqueeze(0) # shape: (1, 1, 3, 3) = (Batch,Channel,H,W)
			else:
				self.kernel = torch.ones([3,3,3],dtype=self.type).unsqueeze(0).unsqueeze(0) # shape: (1, 1, 3, 3, 3) = (Batch,Channel,D,H,W)
			
		elif connect==4:
			if not self.dim3D:
				self.kernel = torch.tensor([
						[0, 1, 0],
						[1, 1, 1],
						[0, 1, 0] ], 
						dtype=self.type
						).unsqueeze(0).unsqueeze(0) # shape: (1, 1, 3, 3) = (Batch,Channel,H,W)
			else:
				self.kernel = torch.tensor([[[0, 1, 0],[1, 1, 1],[0, 1, 0]],
									[[0, 1, 0],[1, 1, 1],[0, 1, 0]],
									[[0, 1, 0],[1, 1, 1],[0, 1, 0]]],dtype=self.type
									).unsqueeze(0).unsqueeze(0) # shape: (1, 1, 3, 3,3) = (Batch,Channel,D,H,W)
		else:
			kernel = None
		
		if not self.dim3D:
			conv = nn.Conv2d(1, 1, kernel_size=self.kernel.shape[-1],
					stride=1, padding=1, bias=False)
		else:
			conv = nn.Conv3d(1, 1, kernel_size=self.kernel.shape[-1],
					stride=1, padding=1, bias=False)

		with torch.no_grad():
			conv.weight = nn.Parameter(self.kernel)
		self.conv = conv.type(self.type).to(self.device)
	
	# pass only img if on same device
	def __call__(self,img):
		# first construct the mask (foreground=1) from the image
		mask = (img>self.background).type(self.type)
		if self.iters>0:
			# repeatedly erode or dilate
			for i in range(self.iters):
				if self.operation=='dilate':
					mask = (self.conv(mask)>0).type(self.type)
				if self.operation=='erode':
					mask = (self.conv(mask)>=self.connectivity).type(self.type)
		# final threshold after erosion or dilation
		return mask
