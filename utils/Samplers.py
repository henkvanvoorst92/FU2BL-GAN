import numpy as np
#all classes below are designd to sample indices to select slices, patches or cubes at random for training

class SliceSampler(object):
    """
    Samples slices from a distribution for 2 modalities
    """
    def __init__(self,sample_func=np.random.randint,
                 paired=True, 
                 margin=(.1,.9), 
                 input_n_slices = 1):
        """
        sample_func: the function used to sample an integer (between margin from z-dim)
        paired: if True the same physical slice in modality 1 and 2 is chosen, 
        else a new slice is drawn for modality 2. Alternatively, if paired=None only a single sample is returned
        margin: margin on the top and bottom of the volume where no sampling is performed
        dim3d: if higher than 0, the number of slices is added to the top and bottom of the sampled integer slice number
        """
        self.sample_func = sample_func
        self.paired = paired
        self.margin = margin
        self.input_n_slices = input_n_slices 

    def sample_integer(self, zdim):
        if isinstance(self.margin, float):
            mrgn = int(self.margin*zdim)
            low, high = mrgn, zdim-mrgn
        elif isinstance(self.margin,int):
            mrgn = self.margin
            low, high = mrgn, zdim-mrgn
        elif isinstance(self.margin, tuple):
            low, high = self.margin
            low, high = int(low*zdim), int(high*zdim)
        
        i = self.sample_func(low,high)
        # retrieve dim3d number slices before and after sampled number i
        if self.input_n_slices>1:
            dim3d = int(np.floor_divide(self.input_n_slices,2))
            out = [i]
            for j in range(dim3d):
                out.append(i-(j+1))
                out.append(i+(j+1))
            out = list(set(out))
        else:
            out = i
        return out

    def __call__(self, zdim1, zdim2=None):
        i1 = self.sample_integer(zdim1)
        if self.paired==True:
            i2 = i1
            return i1, i2
        elif self.paired==False:
            i2 = self.sample_integer(zdim2)
            return i1, i2
        elif self.paired==None:
            return i1

class SliceSamplerTwoStaged(object):
    """
    Samples slices from a distribution for 2 modalities
    """
    def __init__(self,
                 paired=True, 
                 margin1=(.1,.9), 
                 margin2 = (.3,.7),
                 prob_fact = 3,
                 input_n_slices = 1):
        """
        sample_func: the function used to sample an integer (between margin from z-dim)
        paired: if True the same physical slice in modality 1 and 2 is chosen, 
        else a new slice is drawn for modality 2. Alternatively, if paired=None only a single sample is returned
        margin: margin on the top and bottom of the volume where no sampling is performed
        dim3d: if higher than 0, the number of slices is added to the top and bottom of the sampled integer slice number
        """
        self.paired = paired
        self.margin1 = margin1
        self.margin2 = margin2
        self.prob_fact = prob_fact
        self.input_n_slices = input_n_slices 

    def sample_integer(self, zdim):
        slices = np.arange(0,zdim+1)
        pslices = slices/zdim
        p_slices = np.zeros_like(pslices)
        p_slices[(pslices>=self.margin2[0])|(pslices<=self.margin2[1])] = self.prob_fact
        p_slices[(pslices<self.margin2[0])|(pslices>self.margin2[1])] = 1
        p_slices[(pslices<self.margin1[0])|(pslices>self.margin1[1])] = 0
        p_slices = p_slices/(p_slices.sum())
        i = int(np.random.choice(slices, p=p_slices))

        # retrieve dim3d number slices before and after sampled number i
        if self.input_n_slices>1:
            dim3d = int(np.floor_divide(self.input_n_slices,2))
            out = [i]
            for j in range(dim3d):
                out.append(i-(j+1))
                out.append(i+(j+1))
            out = list(set(out)) 
        else:
            out = i

        return out

    def __call__(self, zdim1, zdim2=None):
        i1 = self.sample_integer(zdim1)
        if self.paired==True:
            i2 = i1
            return i1, i2
        elif self.paired==False:
            i2 = self.sample_integer(zdim2)
            return i1, i2
        elif self.paired==None:
            return i1

class ValSliceSampler(object):
    """
    Selects slices from a z_dim with margin
    n_slices is (approximately) the number of slices returned
    This code is used to define validation slices
    """
    def __init__(self,n_slices=10, # 
                 margin=(.1,.9), 
                 input_n_slices = 1):
        """
        n_slices: number of slices per volume repeatedly used for validation
        margin: margin on the top and bottom of the volume where no sampling is performed
        input_n_slices: if higher than 1 (always uneven), slices indices on the top and bottom of the middle are returned
        """
        self.n_slices = n_slices
        self.margin = margin
        self.input_n_slices = input_n_slices        
     
    def slice_selector(self, zdim):
        if isinstance(self.margin, float):
            mrgn = int(self.margin*zdim)
            low, high = mrgn, zdim-mrgn
        elif isinstance(self.margin,int):
            mrgn = self.margin
            low, high = mrgn, zdim-mrgn
        elif isinstance(self.margin, tuple):
            low, high = self.margin
            low, high = int(low*zdim), int(high*zdim)
        # steps to take in range for sampling slices
        steps = int((high-low)/self.n_slices)
        slices = list(range(low,high,steps))

        # retrieve dim3d number slices before and after sampled number i
        if self.input_n_slices>1:
            dim3d = int(np.floor_divide(self.input_n_slices,2))
            out = []
            for i in slices:
                tmp = [i]
                for j in range(dim3d):
                    tmp.append(i-(j+1))
                    tmp.append(i+(j+1))
                tmp = list(set(tmp)) 
                out.append(tmp)
        else:
            out = slices

        return out

    def __call__(self, zdim):
        return self.slice_selector(zdim)


class CubeSampler(object):
    """
    Samples cubes from a distribution for 2 modalities
    """
    def __init__(self,sample_func=np.random.randint,
                 paired=True, 
                 margin=None, cubeshape=(64,64,64)):
        """
        sample_func: the function used to sample an integer (between margin from z-dim)
        paired: if True the same physical slice in modality 1 and 2 is chosen, 
        else a new slice is drawn for modality 2. Alternatively, if paired=None only a single sample is returned
        margin: margin on the top and bottom of the volume where no sampling is performed --> for cubes default None is best
        cubeshape: shape of the cube sampled
        """
        self.sample_func = sample_func
        self.paired = paired
        self.margin = margin
        self.cubeshape = cubeshape

    def sample_coord(self, scanshape):
        #define range per dim from min to max
        max_ixs = np.array(scanshape)#-np.array(self.cubeshape)
        min_ixs = np.array(self.cubeshape)
        coord = []
        # truncate upper and lower dims if margin is given
        # default values are not changed if no margin is given     
        #print(scanshape)   
        for i,dimsize in enumerate(scanshape):
            # truncate upper and lower slices if margin is given
            # default values are not changed if no margin is given
            low,high = 0,scanshape[i]-self.cubeshape[i] #self.cubeshape[i]
            if isinstance(self.margin, float):
                mrgn = int(self.margin*dimsize)
                low, high = low+mrgn, high-mrgn
            elif isinstance(self.margin,int):
                mrgn = self.margin
                low, high = low+mrgn, high-mrgn
            elif self.margin is not None:
                print('error wrong type of margin', self.margin)

            max_ixs[i] = np.clip(max_ixs[i],low,high)
            min_ixs[i] = np.clip(min_ixs[i],low,high)
            #print(min_ixs[i],max_ixs[i])
            if min_ixs[i]>=max_ixs[i]:
                coord.append(max_ixs[i]-10)
                   # raise ValueError('data is not large enough:', 
                   #      scanshape,min_ixs,max_ixs,
                   #      'for cube of:', self.cubeshape,)
            else:
                coord.append(np.random.randint(min_ixs[i],max_ixs[i]))
                #print('coordinates',coord)

        return coord # return the top left coordinate

    def __call__(self, scanshape1, scanshape2=None):
        coord1 = self.sample_coord(scanshape1)
        if self.paired==True:
            coord2 = coord1
            return coord1, coord2
        elif self.paired==False:
            coord2 = self.sample_coord(scanshape2)
            return i1, i2
        elif self.paired==None:
            return coord1

class SliceLabelSampler(object):
    """
    Samples slices from an array of labels (each slice is a label) --> only foreground label
    """
    def __init__(self,
                 label_nos = [1],
                 paired=False, 
                 margin = None,
                 input_n_slices = 1,
                 closeby = None,
                 multiclass_weights = None, # if label passed is one hot vector this weights each class for sampling
                 ):
        """
        label_nos: what (foreground) label values should be included for the sampling
        paired: if True the same physical slice in modality 1 and 2 is chosen, 
        else a new slice is drawn for modality 2. Alternatively, if paired=None only a single sample is returned
        margin: margin on the top and bottom of the volume where no sampling is performed 
        input_n_slices: how many context slices to return
        closeby: if a float is passed the second modality should be within n-slices range of the first (ascertains more similar slices)
        """
        self.paired = paired
        self.label_nos = label_nos
        self.margin = margin
        self.input_n_slices = input_n_slices 
        self.closeby = closeby
        if  isinstance(closeby,float):
            self.use_closeby = True
        else:
            self.use_closeby = False
        self.multiclass_weights = multiclass_weights

    def sample_integer(self, labels):
        # sample from the foreground (class=label_no) labels only 
        # option to have multiple labels in one label array --> specify which to choose
        if self.multiclass_weights is None or len(labels.shape)<2:
            ix_choices = [ix for ix,y in enumerate(labels) if y in self.label_nos]
            i = np.random.choice(ix_choices,size=1)[0]
        else: # use weight for each label based on multiclass_weights
            weighted_labels = np.max((labels*self.multiclass_weights), axis=1)
            weighted_labels = weighted_labels/weighted_labels.sum()
            i = np.random.choice(np.arange(labels.shape[0]),size=1,p=weighted_labels)[0]
        
        # retrieve dim3d number slices before and after sampled number i
        if self.input_n_slices>1:
            dim3d = int(np.floor_divide(self.input_n_slices,2))
            out = [i]
            for j in range(dim3d):
                out.append(i-(j+1))
                out.append(i+(j+1))
            out = list(set(out))
        else:
            out = i
        return out
    
    def sample_closeby(self,ix1, labels1,labels2):
        # samples an integer from labels2 close to ix1
        z1 = labels1.shape[0]
        z2 = labels2.shape[0]
        
        relative_ix1 = ix1/z1 # the %th slice of labels1
        delta_ix2 = self.closeby*z2
        int_ix2 = int(relative_ix1*z2)
        
        # only sample from control between margins
        if isinstance(self.margin, float):
            mrgn = int(self.margin*zdim)
            low, high = mrgn, z2-mrgn
        elif isinstance(self.margin,int):
            mrgn = self.margin
            low, high = mrgn, z2-mrgn
        elif isinstance(self.margin, tuple):
            low, high = self.margin
            low, high = int(low*z2), int(high*z2)
        else:
            low,high = 0,z2
        
        low = int(np.clip(int_ix2-delta_ix2,low,high))
        high = int(np.clip(int_ix2+delta_ix2,low,high))
        if low>=high:
            i = int_ix2
        else:
            i = np.random.randint(low,high)

        if self.input_n_slices>1:
            dim3d = int(np.floor_divide(self.input_n_slices,2))
            out = [i]
            for j in range(dim3d):
                out.append(i-(j+1))
                out.append(i+(j+1))
            out = list(set(out))
        else:
            out = i
        return out

    def __call__(self, labels1, labels2):
        i1 = self.sample_integer(labels1)
        if self.paired==True:
            i2 = i1
            return int(i1), int(i2)
        elif self.paired==False:
            if self.use_closeby:
                i2 = self.sample_closeby(i1,labels1,labels2)
                return int(i1), int(i2)
            else:
                i2 = self.sample_integer(labels2)
                return int(i1), int(i2)
        elif self.paired==None:
            return int(i1)
    


