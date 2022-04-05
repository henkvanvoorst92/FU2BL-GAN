import argparse
from torch import nn

def get_options():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	# dataset parameters #E:\CTA2NCCT\paired_redo\raw\new_21122020
	parser.add_argument('--train_filepath', type=str, default='', help='path to HDF5 training dataset')
	parser.add_argument('--batch_size', type=int, default=2, help='Size of the batch loaded per iteration with the dataloader')
	parser.add_argument('--cache_size', type=int, default=4*8, help='Number of samples loaded in memory to pass to the GPU')
	parser.add_argument('--n_workers', type=int, default=8, help='Number of cpu cores used for loading')
	# model parameters
	parser.add_argument('--G_network', type=str, default='ResNet', help='Network architecture used for generator: ResNet or UNet')
	parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels: 3 for RGB and 1 for grayscale')
	parser.add_argument('--input_n_slices', type=int, default=1, help='# input slices used (depth)')
	parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
	parser.add_argument('--num_downs', type=int, default=7, help='number of downsamples of the generator network')
	parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
	parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')
	parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
	parser.add_argument('--norm', default=nn.InstanceNorm2d, help='instance normalization or batch normalization [instance | batch | none]')
	parser.add_argument('--gan_mode', default='vanilla', help='type of loss defined in GANLoss class |vanilla=BCELogitLoss|lsgan=MSE|wgangp=Wassersteiner GAN loss with gradient penalty|')
	parser.add_argument('--lambda_gp', type=float, default=10.0, help='Lambda used for gradient penalty if wgangp is used (optional)')
	parser.add_argument('--D_mode', type=str, default='relax', help='strict_conditional:Wether or not to use D-input channels 2 (both real_A and fake/real_B or only fake/real_B)')
	parser.add_argument('--background', type=int, default=-1, help='background value after normalization')
	#Augment params
	parser.add_argument('--p_augment', type=float, default=.5, help='prob of standard structural augemntations')
	parser.add_argument('--p_contrast_augment', type=float, default=.2, help='prob of intensity and noise augmentations')
	parser.add_argument('--p_RBN', type=float, default=.2, help='prob of radon beam noise augmentation (beam hardening)')
	parser.add_argument('--factor_RBN', type=tuple, default=(-0.8,0.8), help='factor of RBN added to image')
	parser.add_argument('--only_3D_imgA', type=bool, default=False, help='if true only 3D imgA is loaded imgB is slice')
	parser.add_argument('--seed', type=int, default=21, help='random seed to set for all processes')
	parser.add_argument('--HU_ranges', type=tuple, default=[(0,100),(100,1000)], help='tuple of tuples per input channel range to clip and normalize')
	# train parameters
	parser.add_argument('--G_D_rate', type=int, default=5, help='Number of times the generator is trained for 1 time discriminator training')
	parser.add_argument('--lr', type=float, default=0.00002, help='Learning rate')
	parser.add_argument('--betas', type=tuple, default=(.5, 0.999), help='betas of adam optimizer')
	parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
	parser.add_argument('--masked_L1', type=bool, default=True, help='Use a dilated mask for L1 loss computation (optional)')
	parser.add_argument('--return_brainmask', type=bool, default=False, help='If a mask is available use it else generate mask during training to remove (non skull) background loss')
	parser.add_argument('--masked_GANLoss', type=bool, default=True, help='Use the L1-mask to compute a patch mask for the discriminator loss(optional)')
	parser.add_argument('--mask_itrs', type=int, default=0, help='Amount of erosion/dilation iterations if itrs<0 erode else dilate (optional)')
	parser.add_argument('--print_freq', type=int, default=5, help='Per 5 iters time taken to process is printed')
	parser.add_argument('--loc_checkpoints', type=str, default=r'', help='Path to dir for checkpoint storage')
	parser.add_argument('--label_smoothing', type=float, default=None, help='Smoothing of discriminator labels, None if not used')
	parser.add_argument('--smooth_real_fake', type=tuple, default=(False,False), help='To smooth real or fake both or only one of the two')
	# learning rate related
	# for linear decrease
	parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy for decay. [linear | step | plateau | cosine]')
	parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
	parser.add_argument('--n_epochs_same', type=int, default=500, help='Number of epochs the learning rate is kept equal')
	parser.add_argument('--n_epochs_decay', type=int, default=500, help='number of epochs to linearly decay learning rate to zero')

	opt,__ = parser.parse_known_args()
	return opt