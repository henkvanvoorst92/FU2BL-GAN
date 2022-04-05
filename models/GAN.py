
import torch
import random
import torch.nn as nn
from torch.nn import init
import numpy as np
import functools
from torch.optim import lr_scheduler

# original code from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/pix2pix_model.py

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc=1, output_nc=1,
                 num_downs=7, ngf=64, 
                 norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, depth=1):
        
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. 
                                For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # 
                                at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            depth           -- number of input slices
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure

        # if a depth input (more than 1 slice) first perform conv3d
        self.depth = depth # depth==number of slices
        if self.depth>1:
            n_blocks = np.floor_divide(self.depth,2)
            output_nc_3D = int(ngf/2)
            # use 3D input of surrounding slices
            self.input_model_3D = Input_3D_Module(input_depth=self.depth,
                                        output_nc=output_nc_3D, 
                                        input_nc=1, 
                                        norm_layer=nn.BatchNorm3d, 
                                        use_dropout=True)

            input_nc1 = output_nc_3D
            input_3D = True # adjust input layer input_nc (not equal to output_nc)
        else:
            input_3D, input_nc1 = False, input_nc


        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, 
                            submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, 
                            submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, 
                            submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, 
                            submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, 
                            submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc1, 
                            submodule=unet_block, outermost=True, norm_layer=norm_layer, input_3D=input_3D)  # add the outermost layer
        #print(summary(self.model))
        self.model = self.model.type(torch.float32)

    def forward(self, input):
        """Standard forward"""
        # cast model to similar type as input
        if self.depth>1:
            input = self.input_model_3D(input)

        output = self.model(input)
        return output


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, 
                 norm_layer=nn.BatchNorm2d, use_dropout=False, input_3D=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None and not input_3D:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost: 
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet 
    blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural 
    style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, 
                        depth=1, norm_layer=nn.BatchNorm2d, 
                        use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
            depth (int)         -- number of input slices
        """

        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        # add 3D input block if more than 1 slice is given
        self.depth = depth
        if self.depth>1:
            # use 3D input of surrounding slices
            n_blocks = np.floor_divide(self.depth,2)
            output_nc_3D = int(ngf/2)*input_nc
            # use 3D input of surrounding slices
            self.input_model_3D = Input_3D_Module(input_depth=self.depth, 
                                        output_nc=output_nc_3D, 
                                        input_nc=input_nc, 
                                        norm_layer=nn.BatchNorm3d, 
                                        use_dropout=True)

            input_nc = output_nc_3D # because input of ResNet is output of input_model_3D

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks): # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model).type(torch.float32)

    def forward(self, input):
        """Standard forward"""

        if self.depth>1:
            # process the 3D input to 2D, channels=64, only if >1 slice is given (depth>1)
            input = self.input_model_3D(input)

        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


# 2 types of discriminator: NLayerDiscriminator and PixelDiscriminator

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, ttype=torch.float32):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.ttype = ttype
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, 
            kernel_size=kw, stride=2, 
            padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                    kernel_size=kw, stride=2, 
                    padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
            kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, 
                    kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence).type(self.ttype)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        self.net = self.net.type(input.type())
        return self.net(input)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

##############################################################################
# Classes
##############################################################################


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs_same) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class ImagePool():
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py
    """

    def __init__(self, 
                pool_size, 
                buffer_device='cuda', 
                compute_device='cuda'):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int)     -- the size of image buffer, if pool_size=0, no buffer will be created
            buffer_device(str)  -- device to store buffered data on (preferably 'cuda' if buffer to big 'cpu')
            compute_device(str) -- device to load the buffer data on (preferably 'cuda')
        """
        self.pool_size = pool_size
        self.buffer_device = buffer_device
        self.compute_device = compute_device
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.buffer = [] # use a list object to store historical generated images

    def query(self, data):
        """Return an image from the pool.
        Parameters:
            data: the latest generated images from the generator and corresponding masks 
            (mask=None if not used to remove background loss) list of [img,mask] sets
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return data
        return_fakes, return_reals, return_masks = [], [], []
        for fake_img, real_img, mask in data:
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                
                self.buffer.append([fake_img.clone().to(self.buffer_device),
                                    real_img.clone().to(self.buffer_device), 
                                    mask.clone().to(self.buffer_device)]) # insert new data into the buffer stored on cpu
                
                return_fakes.append(fake_img) # image is already on compute device
                return_reals.append(real_img) # image is already on compute device
                return_masks.append(mask) # mask is already on compute device
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.buffer[random_id]

                    self.buffer[random_id] = [fake_img.clone().to(self.buffer_device),
                                              real_img.clone().to(self.buffer_device), 
                                              mask.clone().to(self.buffer_device)] # insert new data in the buffer on buffer device
                    
                    return_fakes.append(tmp[0].clone().to(self.compute_device)) # img to compute device
                    return_reals.append(tmp[1].clone().to(self.compute_device)) # img to compute device
                    return_masks.append(tmp[2].clone().to(self.compute_device))# mask to compute device

                else:       # by another 50% chance, the buffer will return the current image
                    return_fakes.append(fake_img) # image is already on compute device
                    return_reals.append(real_img) # image is already on compute device
                    return_masks.append(mask) # mask is already on compute device

        out = []
        for img_sets in [return_fakes, return_reals, return_masks]:
            # only return imgs if all are tensors
            if np.all([isinstance(img, torch.Tensor) for img in img_sets]):
                out.append(torch.stack(img_sets, 0))# output dims B,C,D,H,W
            else:
                out.append(None)
        
        return out

class Input_3D_Module(nn.Module):
    """
    Defines the 3D input module prior to a 2D segmentation network
    """

    def __init__(self, input_depth, output_nc, input_nc=1, 
                 norm_layer=nn.BatchNorm3d, use_dropout=True):
        
        """
        -- input_depth (int): z dimension (slices) used as input
        -- first_block_nf (int): number of filters for the first block
        -- output_nc (int): number of output channels of the last block
        -- input_nc (int): number of input channels (RGB=3, greyscale=1)
        -- norm_layer (nn submodule): normalization applied after each conv layer (nn.BatchNorm3d)
        -- use_dropout (bool): including of .5 dropout after each conv layer
        -- device (str): on what device to allocate the model
        """
        super(Input_3D_Module, self).__init__()
        
        self.input_depth = input_depth
        #self.first_block_nf = first_block_nf
        self.output_nc = output_nc
        self.input_nc = input_nc
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout
        
        self.model = self.build_3D_module().type(torch.float32)
        
    def build_3D_module(self):
        """
        Function that constructs the 3D blocks prior to a 2D segmentation network
        3D features are extracted and concatenated with original input depth
        across the channel axis for a subsequent 2D network
        """
        # since the slices are concatenated as channels the 
        # output number of filters has to be reduced
        output_nc = self.output_nc-self.input_depth

        model = []
        tmp_in_nc = self.input_nc
        n_blocks = np.floor_divide(self.input_depth,2)
        for i in range(n_blocks):
            f_scale = 2**(n_blocks-i-1)
            tmp_out_nc = int(f_scale*self.output_nc)
            if i==(n_blocks-1): # the last block is slightly different --> get right output nc
                tmp_out_nc = self.output_nc-self.input_depth*self.input_nc
            
            m = self.conv3D_block(tmp_in_nc, tmp_out_nc, 
                             ksize=(3,3,3), stride=(1,1,1), pad=(0,1,1), 
                             norm=self.norm_layer, 
                             bias=True, 
                             use_dropout=self.use_dropout)
            
            # redefine input and output dimensions
            tmp_in_nc = tmp_out_nc
            model.append(m)

        return nn.Sequential(*model)
    
    @staticmethod 
    def conv3D_block(input_nc,output_nc,
                 stride=(1,1,1),
                 ksize=(3,3,3),
                 pad=(1,1,1),
                 norm=nn.BatchNorm3d, 
                 bias=True, use_dropout=True):
        """" 
        Receives 3D input, uses a kernel in depth dimension equal to 
        the input depth to construct a 2D output with output_nc channels
        """
        model = []
        model.append(nn.Conv3d(input_nc, output_nc, 
                        stride=stride, 
                        padding=pad, 
                        kernel_size=ksize, 
                        bias=bias))

        if not norm==None:
            model.append(norm(output_nc))
        model.append(nn.LeakyReLU(0.2, True))

        if use_dropout:
            model.append(nn.Dropout(0.5))
        return nn.Sequential(*model)
    
    def forward(self,input): # input has shape N,C,D,H,W
        output = self.model(input) # output has shape N,C,D,H,W
        # concat across channel axis --> input depth becomes channel
        if self.input_nc<2:
            output = torch.cat([input.squeeze(1),output.squeeze(2)], dim=1)
        else:
            # dim=1 is the channel dim, stack the input channels
            input = [input[:,i] for i in range(self.input_nc)]
            input = torch.cat(input, dim=1)
            output = torch.cat([input,output.squeeze(2)], dim=1)

        return output
