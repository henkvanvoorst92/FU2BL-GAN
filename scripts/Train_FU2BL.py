### latest update after 16-03-2021 also includes use of WGAN -gp
## by h.vanvoorst@amsterdamumc.nl

import numpy as np

import SimpleITK as sitk
import os, torch, time,sys
sys.path.append(r'C:\Users\hvanvoorst\PhD\git_repos\FU2BL-GAN')
from models.Discriminator import NLayerDiscriminator, PixelDiscriminator
from models.Generator import UnetGenerator,ResnetGenerator
from utils.TrainUtils import init_weights, get_scheduler
from utils.Losses import GANLoss, Masked_L1Loss, weights_init_val, accuracy, Mask_NLayerDiscriminator, cal_gradient_penalty
from utils.cv2_utils import add_text_to_img
from utils.Utils import store_df, rtrn_np

def adjust_learning_rate(optimizer, new_lr):
    for p in optimizer.param_groups:
        p['lr'] = new_lr
    return optimizer

class FU2BL_GAN(object):
    # all below is inspired by: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    def __init__(self, opt, train=True, name='', device=None):
        if device==None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device #

        self.opt = opt

        if self.opt.G_network=='UNet':
        	self.netG = UnetGenerator(input_nc=self.opt.input_nc, output_nc=1,
        		num_downs=self.opt.num_downs,
        		ngf=self.opt.ngf,
        		norm_layer=self.opt.norm,
        		use_dropout=True, depth=self.opt.input_n_slices)

        elif self.opt.G_network=='ResNet':
        	self.netG = ResnetGenerator(input_nc=self.opt.input_nc, output_nc=1,
        		ngf=self.opt.ngf,
	    		norm_layer=self.opt.norm,  
	    		use_dropout=True, 
	    		n_blocks=6, depth=self.opt.input_n_slices)

        self.D_mode = opt.D_mode # wether or not to use 2 channels for D-input
        if self.D_mode=='strict_conditional':
            D_in_nc = 2
        else:
            D_in_nc = 1

        self.name = name
        self.loc_checkpoints = self.opt.loc_checkpoints
        # pm imprement load weights from pretrained model
        if train:
            self.netD = NLayerDiscriminator(input_nc=D_in_nc, 
                    ndf=self.opt.ndf, 
                    n_layers=self.opt.n_layers_D, 
                    norm_layer=self.opt.norm)

            self.train_init()
            
    def train_init(self):
        
        init_weights(self.netG)
        init_weights(self.netD)
        
        self.netG.to(self.device)
        self.netD.to(self.device)
        print('Networks to', self.device)
        
        self.fake_label = False
        self.real_label = True
        
        # initialization for G_D rate required
        self.loss_G, self.loss_D = torch.tensor(100), torch.tensor(100)
        # used for tracking instability (loss to far from min/max values)
        self.global_min_G, self.global_max_D = 100,0
        #self.loss_margin_dct = self.opt.loss_margin_dct
        
        # initialize loss functions
        self.criterionGAN = GANLoss(self.opt.gan_mode, 
                            label_smoothing=self.opt.label_smoothing, 
                            smooth_real_fake=self.opt.smooth_real_fake).to(self.device) #GANloss can be called for computing

        # use a mask to compute L1 loss (reduces background bias of loss)
        if self.opt.masked_L1:
            self.criterionL1 = Masked_L1Loss
        else:
            self.criterionL1 = torch.nn.L1Loss()
            
        if self.opt.masked_GANLoss:
            self.mask_netD = Mask_NLayerDiscriminator(
                                input_nc=1, 
                                n_layers=self.opt.n_layers_D, 
                                padw=1, ksize=4,
                                weights_value=1).to(self.device)
        
        self.learning_rate = self.opt.lr
        # initalize optimizers, pm differentiate training/testing
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.learning_rate, betas=self.opt.betas)

        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.learning_rate, betas=self.opt.betas)

        #self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=self.learning_rate*4, momentum=0.9)
        
        self.label_smoothing = self.opt.label_smoothing
        self.train_output_tracker, self.val_output_tracker = [], []
        self.epoch = 0
        self.iter = 0
        self.errors = []
        
    def set_input(self,imgA, imgB, mask=None):
        """Unpack input data from the dataloader 
        and perform necessary pre-processing steps.

        Default use: imgA=FU, imgB=BL

        Dimensions of data below are Batch,Channel,Depth, Heigth, Width (depth is not required)
        """
        self.Gmask = None
        if self.opt.return_brainmask==False:
            mask = ((imgA[:,0:1]>-.95)*1).type(torch.float32)
        if isinstance(imgA, torch.Tensor):
            if len(imgA.shape)==3 or len(imgA.shape)==4: 
                # optionally add depth dimension (2) if not available and 3D training is performed
                self.depth = 1 # number of slices used
                self.middle_slice_ix = 0

                self.real_A = imgA.to(self.device)
                self.real_B = imgB.to(self.device)

                if self.opt.masked_L1:
                    self.Gmask = mask.to(self.device) # generator mask

            elif len(imgA.shape)==5: # when depth>0 choose the middle slice
                self.depth = imgA.shape[2] # number of slices used
                self.middle_slice_ix = int(np.floor_divide(self.depth,2))
                # Gmask used to compute mask and discriminator mask --> upgraded: mask= A or B == background
                # the masks are used to exclude loss due to background errors
                if self.opt.masked_L1:
                    self.Gmask = mask[:,:,self.middle_slice_ix,:,:]#.to(self.device) # only the middle slice of A is needed to compute mask 

                # only the middle slice of B is needed
                # however in the loader you can also define to only load the middle slice
                if self.depth==imgB.shape[2]:
                    ix = self.middle_slice_ix
                else:
                    ix = 0
                self.real_B = imgB[:,:,ix,:,:].to(self.device)
            else: 
                print('Error shape of input is not good:', imgA.shape, imgB.shape)

        elif isinstance(imgA, list):
            print('LIST is not good')

        # compute Mask than can be used for the discriminator
        if self.opt.masked_L1 or self.opt.masked_GANLoss:
            if self.opt.masked_GANLoss:
                self.Dmask = self.mask_netD(self.Gmask)>0 # Dmask is automatically send to self.device

        del imgA, imgB, mask

    def forward(self):
        """Run forward pass."""
        self.syn_seg = self.netG(self.real_A)  # G(A) --> difference map (syn_seg) 
        real_A = self.real_A

        if len(self.real_A.shape)==5: # self.real_A is 3D pick the middle one
            real_A = real_A[:,:,self.middle_slice_ix,:,:]

        if self.opt.input_nc>1: # only use first input channel dim for computing of fakeB (rest is extra info)
            real_A = real_A[:,:1,:,:]

        self.fake_B = real_A - self.syn_seg

    def backward_D(self, validation=False):
        """Calculate GAN loss for the discriminator"""
        # concatenate across channel axis
        # Fake; stop backprop to the generator by detaching fake_B
        if self.D_mode=='strict_conditional':
            real_A = self.real_A
            if len(self.real_A.shape)==5:
                real_A = real_A[:,:,self.middle_slice_ix,:,:] #dims = N,C,D,H,W
            if self.opt.input_nc>1:
                real_A = real_A[:,:1,:,:]
            D_input = torch.cat((real_A, self.fake_B), 1)
            chix_fake_B = 1  # channel index fake B
        else:
            D_input = self.fake_B
            chix_fake_B = 0  # channel index fake B 

        Dmask = self.Dmask

        pred_fake = self.netD(D_input.detach()) # use detach to not compute Generator gradients
        self.loss_D_fake = self.criterionGAN(pred_fake, self.fake_label, mask=Dmask)
        
        # Real
        if self.D_mode=='strict_conditional':
            D_input = torch.cat((real_A, self.real_B), 1)
        else:
            D_input = self.real_B

        pred_real = self.netD(D_input)
        self.loss_D_real = self.criterionGAN(pred_real, self.real_label, mask=self.Dmask)
        
        if self.opt.gan_mode=='wgangp':
            # compute the gradient penalty from GAN.py file (prevents exploding/vanishing gradients)
            self.gp, self.gradients = cal_gradient_penalty(self.netD, 
                                                            real_B, 
                                                            D_input[:,chix_fake_B].unsqueeze(1), # fake_B
                                                            self.device, 
                                                            lambda_gp=self.opt.lambda_gp)

            self.loss_D = self.loss_D_fake + self.loss_D_real + self.gp

        else:
            # combine loss and calculate gradients (no gradients in validation mode)
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.gp, self.gradients = None, None

        # compute accuracy
        acc_samples = accuracy(pred_fake, pred_real, self.Dmask)
        self.acc_fake, self.acc_real, self.acc_tot = np.array(acc_samples).mean(axis=0)

        del acc_samples, pred_fake, pred_real, D_input

        if not validation:
            self.loss_D.backward(retain_graph=True)

    def backward_G(self, validation=False):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator 

        if self.D_mode=='strict_conditional': # strict conditional implies stacking real_A with fake_B (or real_B)
            real_A = self.real_A
            if len(self.real_A.shape)==5:
                real_A = real_A[:,:,self.middle_slice_ix,:,:] #dims = N,C,D,H,W
            if self.opt.input_nc>1:
                real_A = real_A[:,:1,:,:]
            D_input = torch.cat((real_A, self.fake_B), 1)
            chix_fake_B = 1  # channel index fake B
        else:
            D_input = self.fake_B
            chix_fake_B = 0  # channel index fake B 
            
        pred_fake = self.netD(D_input)
        # compute loss
        self.loss_G_GAN = self.criterionGAN(pred_fake, self.real_label, self.Dmask) #pm add label smoothing
        if not isinstance(self.Gmask, torch.Tensor): # if no mask is available, compute L1 without mask
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 #self.fake_B-1
        else: # compute L1 with mask if mask is available
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B, self.Gmask) * self.opt.lambda_L1 #self.fake_B-1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        
        del pred_fake, D_input

        if not validation:
            self.loss_G.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad                 
    
    def optimize_G(self, incl_forward = True):
        # pm gradient clipping!
        if incl_forward:
            self.forward()     # compute fake images: G(A)
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
    
    def optimize_D(self):
        # pm gradient clipping!
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        
    def optimize_parameters(self, optimize_D=True, optimize_G=True, n_D = 1):
        # if G is not optimized fix gradients to make computing more efficient for D
        # with the exception: when buffering the gradients of net G are required;
        # gradients of fake_B (might be) required after loading from buffer --> pm: only optimize G on current data
        if optimize_G: #  
            self.set_requires_grad(self.netG, True)
        else:
            self.set_requires_grad(self.netG, False)         

        self.forward()
        if optimize_D: 
            for i in range(n_D):
                self.optimize_D() # train D
        else: # only compute metrics in validation mode
            self.set_requires_grad(self.netD, False)
            self.backward_D(validation=True)

        if optimize_G:
       	    self.optimize_G(incl_forward=False) # train G
        else:# only compute metrics in validation mode
            self.backward_G(validation=True)
        
    def invert_labels(self):
        prev_fake, prev_real = self.fake_label, self.real_label
        self.fake_label = prev_real
        self.real_label = prev_fake

    # function to create checkpoint with possibility to resume training
    def save_checkpoint(self): 

        checkpoint = {
            'epoch': self.epoch,
            'iter':self.iter,
            'netG_state_dict': self.netG.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'optG_state_dict': self.optimizer_G.state_dict(),
            'optD_state_dict': self.optimizer_D.state_dict()
        }

        torch.save(checkpoint, os.path.join(self.loc_checkpoints, 
                   'epoch_'+str(self.epoch)+'_'+self.name+'_checkpoint.pth'))
        
        # store visualisation of a selection of validation set results
        
        del checkpoint
        
    def load_checkpoint(self, epoch, model_state='eval'):
        filepath = os.path.join(self.opt.loc_checkpoints,'epoch_'+str(epoch)+ \
                    '_'+self.name+'_checkpoint.pth')
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        if model_state=='train':
            self.netG.train()
            self.netD.load_state_dict(checkpoint['netD_state_dict'])
            self.netD.train()
            self.netD.to(self.device)
            
            self.optimizer_G.load_state_dict(checkpoint['optG_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optD_state_dict'])

            self.epoch = epoch+1
            self.iter = checkpoint['iter']
            
        else:
            # use only netG for inference in evaluation mode
            self.set_requires_grad(self.netG, False) # no grad reduces memory useage
            self.netG.eval() # ascertain fixed eval mode for batchnorm and dropout

        self.netG.to(self.device)

        print('Model Checkpoint at epoch', epoch, 'is loaded on', self.device)

    def at_iter_end(self,epoch,itr,RID, train_G, train_D, 
        t=None, save=True, val_loader=None, epoch_end=False):
        # update iter and epoch in class
        self.epoch = epoch
        self.iter = itr

        self.train_output_tracker.append([epoch,itr,
          self.loss_G.item(),self.loss_G_GAN.item(), 
          self.loss_G_L1.item(), self.loss_D.item(), 
          self.loss_D_real.item(), self.loss_D_fake.item(), 
          #self.gp.item(), self.gradients.mean().item(),
          self.acc_fake, self.acc_real, self.acc_tot, train_G, train_D, t])

        if save and epoch_end and epoch%10==0: #and epoch!=0
            # save a checkpoint at the end of every epoch
            self.save_checkpoint()
            # store training data to a dataframe
            store_df(self.train_output_tracker, os.path.join(self.loc_checkpoints,'train_loss'), '.pic')
            self.train_output_tracker = []

            # optionally also store validation results
            if val_loader!=None:
            	t1 = time.time()
            	self.validation_results(val_loader, save=save, stamp_image=True)
            	t2 = time.time()
            	print('Epoch', epoch, 'validation time:', round(t2-t1))

    def iter_mean_add(self, n_iters_in_epoch):
        # per epoch as a mean value per epoch for D_loss, G_loss, fake_acc, real_acc
        mean_D_loss = self.loss_D.cpu().detach()/n_iters_in_epoch
        mean_G_loss = self.loss_G.cpu().detach()/n_iters_in_epoch
        mean_fake_acc = self.acc_fake/n_iters_in_epoch
        mean_real_acc = self.acc_real/n_iters_in_epoch
        return np.array([mean_D_loss, mean_G_loss, mean_fake_acc, mean_real_acc]).astype('float32')

    def validation_run(self):
        # set_input has to be called before this function is used

        # net G and D are not optimized in validation mode
        self.set_requires_grad(self.netG, False)  
        #self.set_requires_grad(self.netD, False) # it is optional to compute loss D 
        self.forward()

        self.label_smoothing = None
        # compute losses by calling backward in validation mode
        self.backward_G(validation=True)
        self.backward_D(validation=True)

        # store metrics in validation tracker per patient 
        self.val_output_tracker.append([self.RID, self.epoch, self.iter,
                  self.loss_G.item(),self.loss_G_GAN.item(), 
                  self.loss_G_L1.item(), self.loss_D.item(), 
                  self.loss_D_real.item(), self.loss_D_fake.item(),
                  self.acc_fake, self.acc_real, self.acc_tot
                  ])

        self.label_smoothing = self.opt.label_smoothing
        # retrieve np arrays for additional perocessing of images

        # change stuff here
        fake_imgB = rtrn_np(self.fake_B[:,0,:,:]) #-1
        syn_seg = rtrn_np(self.syn_seg[:,0,:,:])

        return fake_imgB, syn_seg

    def validation_results(self, val_loader, save=True, stamp_image=False, batch_store=10):
        """
        Computes validation set results at the end of an epoch
        """

        batch_imgB, batch_synseg, c = [],[], 0
        org_A, org_B = [], []
        for imgA,imgB,RID in val_loader:
            if c>batch_store:
                break

            if isinstance(imgA, np.ndarray):
                imgA = torch.from_numpy(imgA)
            elif isinstance(imgA, torch.Tensor):
                imgA, imgB = imgA[0],imgB[0]
            elif isinstance(imgA, list):
                imgA, imgB = torch.cat(imgA,0), torch.cat(imgB,0)
            else:
                print('Error type of imgA is not compatible:', type(imgA))

            # prepare validation data for predicting
            self.set_input(imgA, imgB) # collaps the batchno dimension (already 11 slices in mem)
            self.RID = RID[0]
            fake_imgB, syn_seg = self.validation_run()

            # at the first epoch a storage of original images is made
            # to compare the generated images in simple ITK with
            if self.epoch==0:
                if self.depth>1: # for 3D used of imgA
                    imgA = rtrn_np(self.real_A[:,0,self.middle_slice_ix,:,:]) 
                else:
                    imgA = rtrn_np(self.real_A[:,self.middle_slice_ix,:,:])
                imgB = rtrn_np(self.real_B[:,0,:,:])

            if stamp_image or batch_store!=None:
                # stamp the RID to the slices for identification
                for j in range(fake_imgB.shape[0]):
                    fake_imgB[j,:,:] = add_text_to_img(fake_imgB[j,:,:],self.RID)
                    syn_seg[j,:,:] = add_text_to_img(syn_seg[j,:,:],self.RID)
                    if self.epoch==0:
                        imgA[j,:,:] = add_text_to_img(imgA[j,:,:],self.RID)
                        imgB[j,:,:] = add_text_to_img(imgB[j,:,:],self.RID)
                            
            if batch_store!=None:
                batch_imgB.append(fake_imgB)
                batch_synseg.append(syn_seg)
                if self.epoch==0:
                    org_A.append(imgA)
                    org_B.append(imgB)
                    
            if save and batch_store==None:
                # store each image separate
                self.save_nifti(fake_imgB, ext=RID[0]+'-synNCCT.nii.gz')
                self.save_nifti(syn_seg, ext=RID[0]+'-synSeg.nii.gz')
                
        # store batch at end of data processing
        if save and batch_store!=None:
            self.save_nifti(np.vstack(batch_imgB), ext='-synNCCT.nii.gz')
            self.save_nifti(np.vstack(batch_synseg), ext='-synSeg.nii.gz')
            if self.epoch==0:
                self.save_nifti(np.vstack(org_A), ext='-CTA.nii.gz')
                self.save_nifti(np.vstack(org_B), ext='-NCCT.nii.gz')
        
        store_df(self.val_output_tracker, os.path.join(self.loc_checkpoints,'val_loss'), '.pic')
        self.val_output_tracker = []
        # Reset gradiant computation of networks
        self.set_requires_grad(self.netG, True)  
        self.set_requires_grad(self.netD, True)

    def save_nifti(self,volume, ext='.nii.gz'):
    	if isinstance(volume, torch.Tensor):
    		volume = rtrn_np(volume)
    	img = sitk.GetImageFromArray(volume)

    	p = os.path.join(self.loc_checkpoints,'epoch_'+str(self.epoch))
    	f = 'ep'+str(self.epoch)+'-'+ext
    	if not os.path.exists(p):
    		os.mkdir(p)

    	sitk.WriteImage(img, os.path.join(p,f))


    def VisRow(self,batch_ix,img_batch,
                    channel=0, type='float32', segmentations=False):
        # visualize (plot) orgCTA, orgNCCT, synNCCT, orgCTA-orgNCCT, orgCTA-synNCCT inline
        out = []
        for img in img_batch:
            if len(img.shape)==5:
                img = img[:,:,self.middle_slice_ix,:,:]
            out.append(rtrn_np(img[batch_ix,channel,:,:]))

        if not segmentations:
            return np.hstack(out)
        else:
            seg1 = self.segmentr(A,B,mask=((A>-1)&(B>-1))).astype(type)
            seg1[seg1==0] = -1
            seg2 = self.segmentr(A,fB,mask=((A>-1)&(fB>-1))).astype(type)
            seg2[seg2==0] = -1
            out.extend([seg1,seg2])
            #seg3 = (A+1)-(fB+1)
            #seg3[seg3==0] = -1
            return np.hstack(out)

