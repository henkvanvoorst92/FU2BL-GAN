
import torch
import torch.nn as nn
import numpy as np
import functools


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

