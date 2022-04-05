from __future__ import division
import torch, math, random, numbers, os, sys, h5py
import numpy as np
import SimpleITK as sitk

#### TRAIN LOADER functions #########
class Compose(object):
    """
    Alternative to normal Compose class: keeps track of image type
    if a mask is passed and not all changes are the same --> change mask type (float32)

    Composes several transforms (augmentations) together.
    Args:
    transforms (List[Transform]): list of transforms to compose.
    Example:
    >>> transforms.Compose([
    >>>   transforms.CenterCrop(10),
    >>>   transforms.ToTensor(),
    >>> ])
    """
    def __init__(self, transforms, type='float32'):
        self.transforms = transforms
        self.type = type

    def __call__(self, imgs):
        for c,t in enumerate(self.transforms):
            imgs = [img.astype(self.type) for img in imgs]
            imgs = t(imgs)
            #print([type(img[0,0]) for img in imgs])
        return [img.astype(self.type) for img in imgs]



class HDF5Dataset(torch.utils.data.Dataset):
	"""
	Loader for FU2BL-GAN, returns BL and FU scan with ID as iterable dataset
	Loads from a HDF5 dataset with ID - BL,FU
	Also adds radon beam hardening noise (RBN) if required, 
	RBN is stored in the dataset as computation on the fly is to demanding
	
	h5py has the main advantage that lazy loading of slices or patches can be performed (=computationally efficient)

	all is inspired by: https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
	"""
	def __init__(self, 
				 file_path, # path to H5py dataset
				 modalities=['BL', 'FU'], # modalities to load should correspond to subdir name of ID in h5py
				 data_cache_size = 4, # number of scans/slices in cache
				 slice_sampler = None, # a sampling function for sampling slices in train mode
				 transform=None, # transform classes for CTA-NCCT scans (augmentation)
				 mode='train', # train or validation mode
				 val_slice_selector = None, # in validation mode a function to select slices
				 prob_RBN = .2, # probability of adding beam hardening noise
				 factor_RBN = (.1,1.5), # factor of RBN added
				 allow_single_BL = True, #
				 default_dims = (512,512), # default x-y dimension for slices
				 output_background=-1, # background value if volumes are returned
				 output_type ='int16', # type of the output array values
				 background=-1024,
				 return_brainmask=False):  # return brainmask if available in dataset --> not available in RBN dataset

		super().__init__()
		# general inputs
		self.file_path = file_path
		self.m1,self.m2 = modalities
		self.val_slice_selector = val_slice_selector
		self.allow_single_BL = allow_single_BL
		if self.val_slice_selector!=None:
			self.slice_dict = {}
		self.data_info = []
		self.data_cache = {}
		self.data_cache_size = data_cache_size

		self.d1, self.d2 = default_dims
		self.output_background = output_background
		self.output_type = output_type

		# data sampling inputs
		self.slice_sampler = slice_sampler
		self.transform = transform
		self.mode = mode
		self.return_brainmask = return_brainmask
		self.background = background
		self.prob_RBN = prob_RBN
		self.factor_RBN = factor_RBN
		# initialize list of data samples
		self._init_data_infos()

	# store information of each sample in the dataset
	def _init_data_infos(self,): 
		# initialize a list of dictionaries (self.data_info) to obtain data from
		with h5py.File(self.file_path, 'r') as h5_file:
			idx = -1
			for ID, maindr in h5_file.items():
				# store data in dictionary per pt
				info_dct = {'ID':ID, 'cache_idx':idx} 
				fu_times = list(maindr.keys())
				samples = []
				for fu_time in fu_times:
					bl_path = maindr[fu_time]['BL'].name
					fu_path = maindr[fu_time]['FU'].name
					rbn_path = maindr[fu_time]['RBN'].name
					scanshape  = maindr[fu_time]['BL'].shape
					samples.append([bl_path,fu_path, rbn_path, scanshape, fu_time]) # pm: use samples as dictionary
						
				info_dct['data'] = samples
				self.data_info.append(info_dct)
			h5_file.close()
			
	def __len__(self):
		return len(self.data_info)

	def __getitem__(self, index):
		# get data if called with index from data_infos
		# then apply transforms (augmentations)

		data = self.get_data(index) # pm change data loader to fetch more slices
		if not self.return_brainmask:
			img_A, img_B = data       
		else:
			img_A, img_B, mask = data

		if self.transform is not None: # and self.mode=='train'
			# NOTE preprocessing for each pair of images
			# this has to occur joint for paired result
			if isinstance(img_A, np.ndarray):
				# single transfomration of 2 loaded np array
				#if not self.has_mask:
				if not self.return_brainmask:
					(img_A, img_B)  = self.transform([img_A, img_B])
				else:
					(img_A, img_B, mask)  = self.transform([img_A, img_B, mask])
				
			# if img_A (and B) are a list, process the list with images
			elif isinstance(img_A, list):
				if isinstance(img_A[0], np.ndarray):
					# returns a list with both imgs A and B
					if not self.return_brainmask:
						imgs  = self.transform([*img_A, *img_B])
					else:
						imgs  = self.transform([*img_A, *img_B, *mask])
						mask = np.stack(imgs[int(len(img_A)+len(img_B)):])
					# stack both img A and B to get single volume
					img_A = np.stack(imgs[:int(len(img_A))])
					img_B = np.stack(imgs[int(len(img_A)):int(len(img_A)+len(img_B))])
					
				elif isinstance(img_A[0], list):
					outA, outB, out_mask = [], [], []
					for i in range(len(img_A)):
						# returns a list with both imgs A and B
						if not self.return_brainmask:                   
							imgs  = self.transform([*img_A[i], *img_B[i]])
						else:
							imgs  = self.transform([*img_A[i], *img_B[i], *mask[i]])
							out_mask.append(np.stack(imgs[int(len(img_A)+len(img_B)):]))
							
						outA.append(np.stack(imgs[:int(len(img_A[0]))]))
						outB.append(np.stack(imgs[int(len(img_A)):int(len(img_A)+len(img_B))]))
					img_A, img_B, mask = outA, outB, out_mask
					
				else:
					print('Error img type not suitable for train transforms:', 
						type(img_A[0]), type(img_B[0]))
			else:
				print('Error img type not suitable:', 
						type(img_A), type(img_B))
		
		if not self.return_brainmask:
			return img_A, img_B, self.ID
		else:
			return img_A, img_B, mask, self.ID

	def get_data(self, i):
		"""
		Call this function anytime you want to access a chunk of data from the
		dataset. This will make sure that the data is loaded in case it is
		not part of the data cache (=already in memory).
		"""
		data = self.data_info[i]['data']
		if len(data)==1: # if one paired FU-BL data sample available --> pick it
			p_BL, p_FU, p_rbn, shape, self.fu_moment = data[0]
		else: # sample one of the available FU-BL pairs at random
			ix = np.random.randint(len(data))
			p_BL, p_FU, p_rbn, shape, self.fu_moment = data[ix]
			
		self.ID = self.data_info[i]['ID']
		if self.ID not in self.data_cache.keys():
			#print(self.ID)
			self._load_data(p_BL,p_FU,p_rbn,self.ID)
			return self.data_cache[self.ID]

	def _load_data(self, p1,p2,p_rbn,ID):
		"""Load data to the cache given the file
		path and update the cache index in the
		data_info structure.
		"""
		# pm add code for using the mask
		with h5py.File(self.file_path, 'r') as h5_file:
			# load both scans (CTA,NCCT) from the paths given
			# use a slice sampler if given in the class
			scan1 = h5_file[p1] #BL
			scan2 = h5_file[p2] #FU
			rbn = h5_file[p_rbn]
			
			#initialize radon beam hardening application
			flag_RBN = random.random() < self.prob_RBN
			if isinstance(self.factor_RBN,tuple):
				factor = 0
				while abs(factor)<.2: # only use a factor that is larger than 0.2
					factor = round(random.uniform(*self.factor_RBN),2)

			else:
				factor = self.factor_RBN

			if self.mode=='train':
				if self.slice_sampler==None: 
					# if no slice sampler return volume
					s1 = scan1[:,:,:]
					s2 = scan2[:,:,:]
					#if self.has_mask:
					if flag_RBN:
						RBN = rbn[:,:,:].astype(type(s1[0,0,0]))
				elif self.slice_sampler!=None:
					# sample slices to return 
					z1, z2 = scan1.shape[0], scan2.shape[0]
					a_slice, b_slice = self.slice_sampler(z1,z2)
					#print(a_slice,b_slice)
					if isinstance(a_slice,int):
						#sample single slice 
						s1, s2 = scan1[a_slice,:,:], scan2[b_slice,:,:]
						#if self.has_mask:
						if flag_RBN:
							RBN = rbn[a_slice,:,:].astype(type(s1[0,0]))
							s1 = self.add_RBN(s1,RBN,factor, 250)
							s2 = self.add_RBN(s2,RBN,factor, 250)

						### Sanity check of slices sampled ###
						#check if nan in slice, if min==max, or if max is below clipping threshold (normalization does not work then)
						# this problem does not occur with entire scans since these will (almost always)
						# contain both fore and background (more variation in pixel values)
						c = 0
						poor_scan = True
						while poor_scan:
							# resample slice while any contains nan
							a_slice, b_slice = self.slice_sampler(z1,z2)
							s1, s2 = scan1[a_slice,:,:], scan2[b_slice,:,:]
							if flag_RBN:
								RBN = rbn[a_slice,:,:].astype(type(s1[0,0]))
								s1 = self.add_RBN(s1,RBN,factor, 250)
								s2 = self.add_RBN(s2,RBN,factor, 250)

							c+=1
							m = ((s1<300)&(s1>0)).sum()
							# sanity check of the data quality (no nan not only background, min HU<max HU)
							if self.return_brainmask:
								poor_scan = (np.isnan(s1).any()) or (np.isnan(s2).any())\
								or (s1.min()==s1.max()) or (s2.min()==s2.max())\
								or (s1.max()<=-100) or (s2.max()<=-100) or (mask.sum()<100) or m<10000
							else:
								poor_scan = (np.isnan(s1).any()) or (np.isnan(s2).any())\
								or (s1.min()==s1.max()) or (s2.min()==s2.max())\
								or (s1.max()<=-100) or (s2.max()<=-100) or m<10000

							if c==600:
								print('Error: no slice without nan for', ID)
								break

					# !!! caution with multi slice samplers as there may still be errors 
					# (np.NaN in image) in the samples after transforms (clip, normalize)
					elif isinstance(a_slice,list):
					#	if self.only_3D_imgA:
					#		ix = int(np.floor_divide(len(b_slice),2))
					#		middle_sliceno = b_slice[ix]
						s1, s2 = [], []
						#if self.has_mask:
						if self.return_brainmask:
							mask = []
						for a,b in zip(a_slice, b_slice):
							s1.append(scan1[a,:,:])
							#if self.only_3D_imgA:
							#	if middle_sliceno==b:
							#		s2.append(scan2[b,:,:])
							#else:
							s2.append(scan2[b,:,:])
							#if self.has_mask:
							if self.return_brainmask:
								mask.append(MASK[a,:,:])

					else:
						print('Error wrong data type for a_slice, b_slice (sampled slices):', 
						type(a_slice), type(b_slice))
					del z1
					del z2 
				else:
					print('Error no slice sampler defined')
			else:
				print('Error mode is not properly defined as validation or train:', self.mode)

			

			# attach data to cache dictionary
			if self.return_brainmask:
				idx = self._add_to_cache((s1,s2,mask), ID)
			else:
				idx = self._add_to_cache((s1,s2), ID)
			# find index in data info of loaded scan and update cache_idx
			i = [c for c,di in enumerate(self.data_info) if di['ID']==ID][0]
			self.data_info[i]['cache_idx'] = idx
			del scan1, scan2
			h5_file.close()

		# if cache is full remove first in and reset cache_idx
		if len(self.data_cache) > self.data_cache_size:
			removal_keys = list(self.data_cache)
			# remove most recent added from remove list
			removal_keys.remove(ID)
			# remove first in (or random first key)
			#del self.data_cache[removal_keys[0]]
			self.data_cache.pop(removal_keys[0])
			# reset cache index
			self.reset_cache_idx(removal_keys[0])

	# optimized     
	def _add_to_cache(self, data, ID):
		"""
		Passes data (both modalities) and puts it 
		in data_cache; a dict of key=ID:value=(scan1,scan2)
		"""
		if ID not in self.data_cache: 
			self.data_cache[ID] = data

		return len(self.data_cache) - 1

	def reset_cache_idx(self,reset_ID):
		"""
		Resets the cache index after reset_ID is removed
		from the cache, also resets the indices 
		of remaining cache IDs chronologically
		"""
		cache_keys = list(self.data_cache.keys())# ids that remain in cache
		out = []
		for di in self.data_info:
			if di['ID'] == reset_ID:
				di.update({'cache_idx':-1})
			elif di['ID'] in cache_keys:
				di.update({'cache_idx':cache_keys.index(di['ID'])})
			out.append(di)
		self.data_info = out

	def add_RBN(self,img,noise,factor, max_HU_brain):
	    #adds RBN to image and adjust shift in median HU
	    mask = ((img>self.background)&(img<max_HU_brain))
	    pre_median = np.median(img[mask].flatten()) 

	    img = (img+noise*factor*(mask*1)) #only add noise inside the mask
	    post_median = np.median(img[mask].flatten())
	    # Adjust median HU values
	    delta = (np.zeros_like(img)+post_median-pre_median)*(mask*1)
	    img-= delta
	    return img
