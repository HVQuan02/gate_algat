import os
import torch
import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--seed', type=int, default=2024, help='seed for randomness')
        parser.add_argument('--gcn_layers', type=int, default=2, help='number of gcn layers')
        parser.add_argument('--dataset', default='cufed', choices=['pec', 'cufed'])
        parser.add_argument('--dataset_root', default='/kaggle/input/thesis-cufed/CUFED', help='dataset root directory')
        parser.add_argument('--feats_dir', default='/kaggle/input/mask-cufed-feats', help='global and local features directory')
        parser.add_argument('--split_dir', default='/kaggle/input/cufed-full-split', help='train split and val split')
        parser.add_argument('--save_dir', default='weights', help='directory to save checkpoints')
        parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
        parser.add_argument('--t_step', nargs="+", type=int, default=[3, 5, 7, 9, 13], help='Classifier frames')
        parser.add_argument('--input_size', type=int, default=224, help='expected input image size')
        parser.add_argument('--threshold', type=float, default=0.75, help='threshold for logits to labels')
        parser.add_argument('-v', '--verbose', action='store_true', help='show details')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='GCN Album Classification')
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def parse(self):
        opt = self.gather_options()

        self.print_options(opt)

        # set gpu ids
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt