import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True,  help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')#
        #self.parser.add_argument('--loadSize2', type=int, default=86, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        # self.parser.add_argument('--fineSize2', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resvit', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='resvit_many',
                                 help='chooses which model to use. resvit_many, resvit_one, or test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')#图像形状
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--vit_name', type=str, default='Res-ViT-B_16',help='vit type')
        self.parser.add_argument('--pre_trained_path', type=str,default='/media/data/wxd/Checkpoints/le_re_pre/latest_net_G.pth',help='path to the pre-trained resnet architecture')
        self.parser.add_argument('--pre_trained_transformer', type=int, default=1,help='Pre-trained ViT or not')
        self.parser.add_argument('--pre_trained_resnet', type=int, default=0,help='Pre-trained residual CNNs or not')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

