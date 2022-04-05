
import torch
import random
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools

# original code: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0, 
        label_smoothing=None, smooth_real_fake=(True,True), minmax_loss = (-1,1), loss_weights=None):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. 
                               It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        self.loss_weights = loss_weights # weights of each class (only if multiclass used)
        self.label_smoothing = label_smoothing
        self.smooth_real, self.smooth_fake = smooth_real_fake

        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss(weight=self.loss_weights, reduction='none')
            self.masked_loss = nn.MSELoss(weight=self.loss_weights, reduction='none')
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss(weight=self.loss_weights, reduction='mean')
            self.masked_loss = nn.BCEWithLogitsLoss(weight=self.loss_weights, reduction='mean')
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """
       # obtain target label value (0 or 1)

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        # important to cast output to prediction type
        # target tensor is set to have same dims as prediction (only zeros/ones same shape as prediction)
        target_tensor = target_tensor.expand_as(prediction).type(prediction.type()) 

        if self.label_smoothing!=None: # create a sampler to sample smoot labels (not 1 but 0.9)
            if (self.real_label==1) and self.smooth_real:
                smplr = torch.distributions.Uniform(target_tensor-self.label_smoothing, target_tensor) 
            elif (self.fake_label==0) and self.smooth_fake: # create a sampler to sample smoot labels (not 0 but 0.1)
                smplr = torch.distributions.Uniform(target_tensor+self.label_smoothing, target_tensor)
            target_tensor = smplr.sample().type(prediction.type()) 

        return target_tensor

    def __call__(self, prediction, target_is_real, mask = None):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
            label_mask - - a mask of the same shape of prediction and target that is used to
                            ignore part of receptive field in the computation of the loss
            label_smoothing -- a number between 0 and 0.5 to draw a label from an uniform distribution
                                this is used to smooth the label for training the discriminator
            smooth_real_fake -- can be changed to only smooth real or fakes

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            # target tensor has same dims as prediction

            # target_is_real could be set as a one hot encoded label tensor to let discriminator use multiclass classification
            if isinstance(target_is_real, torch.Tensor):
                target_tensor = target_is_real.unsqueeze(-1).unsqueeze(-1).expand_as(prediction).type(prediction.type())
            else:
                target_tensor = self.get_target_tensor(prediction, target_is_real)

            # compute loss
            if isinstance(mask, torch.Tensor):
                # use label_mask to ignore background patches
                loss = (self.masked_loss(prediction, target_tensor)*mask.type(prediction.type())).mean()
            else:
                loss = self.loss(prediction, target_tensor).mean()

            del target_tensor

        elif self.gan_mode == 'wgangp':
            #https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
            # scores of fakes should be >0, scores of reals should be <0
            # for generator: fake should have score <0
            if target_is_real: # reals are positive loss (if prediction >0 then wrong; fakes are>0)
                loss = prediction #*-1 #WGAN_2
            else: # fakes result in negative loss 
                loss = prediction*-1 #WGAN_1
                # baumgartner multiplies fake with -1

            if isinstance(mask, torch.Tensor):
                loss = (loss*mask.type(prediction.type()))
            else:
                loss = loss.mean()
        
        #del prediction

        return loss

# Used for custom loss functions
def Masked_L1Loss(ytrue,ypred,mask):
    return (torch.abs(ytrue-ypred)*mask).mean()

def weights_init_val(m, val=1):
    if isinstance(m, nn.Conv2d):
        m.weight.data.fill_(val)

# accuracyof discriminator output
def accuracy(pred_fake,pred_real,mask=None):
    # inputs have shapes (batchsize,channel,x,y)
    
    # first use a sigmoid to retrieve prediction
    pred_fake, pred_real = torch.sigmoid(pred_fake), torch.sigmoid(pred_real)
    
    # total number of patches equals last two dimensions if mask is not used
    if not isinstance(mask,torch.Tensor):
        n_tot = (pred_fake.shape[-1]*pred_fake.shape[-2])
        
    out = []
    for i in range(pred_fake.shape[0]):
        fake_pred_c = (pred_fake[i,:,:,:]<.5) # n fakes predicted as fakes
        real_pred_c = (pred_fake[i,:,:,:]>.5) # n reals predicted as reals
        if isinstance(mask,torch.Tensor):
            # only foreground of mask is used for the total patch count
            n_tot = mask[i,:,:,:].sum().item()+1e-8        
            fake_pred_c = fake_pred_c[mask[i,:,:]] # n fakes in mask predicted as fake
            real_pred_c = real_pred_c[mask[i,:,:]] # n reals in mask predicted as real
          
        acc_fake = (fake_pred_c.sum().item()+1e-8)/n_tot
        acc_real = (real_pred_c.sum().item()+1e-8)/n_tot
        out.append([acc_fake, acc_real, (acc_fake+acc_real)/2])
    # compute batch mean of the accuracy
    return out

def Mask_NLayerDiscriminator(input_nc=1, n_layers=3, 
                             padw=1, ksize=3,
                             weights_value=1):
    """
    Creates a Discriminator network with only Conv2d layers
    with weights initialized as ones. 
    This network can be used to pass a mask forward and compute
    the receptive field mask that can be used to adjust the 
    discriminator loss (add attention to the foreground patches)
    """
    
    layer_list = [nn.Conv2d(input_nc, 1, 
            kernel_size=ksize, stride=2, 
            padding=padw, bias=False)]
    for n in range(1, n_layers):
        layer_list.append(nn.Conv2d(input_nc, 1, 
            kernel_size=ksize, stride=2, 
            padding=padw, bias=False))
    
    final_lyr = nn.Conv2d(input_nc, 1, 
            kernel_size=ksize, stride=1, 
            padding=padw, bias=False)
    layer_list.extend([final_lyr,final_lyr])
    
    mask_Dnet = nn.Sequential(*layer_list)
    mask_Dnet.apply(weights_init_val)
    return mask_Dnet.eval()



####
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss

    source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/c028d9f64c64ba50b64348129a6ed2eefdfaffee/models/networks.py#L271-L306
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None



