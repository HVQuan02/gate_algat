from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('vigat_model', nargs=1, help='Frame trained model')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
        parser.add_argument('--milestones', nargs="+", type=int, default=[16, 35], help='milestones of learning decay')
        parser.add_argument('--num_epochs', type=int, default=40, help='number of epochs to train')
        parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        parser.add_argument('--resume', default=None, help='checkpoint to resume training')
        parser.add_argument('--beta', type=float, default=1e-6, help='Multiplier of gating loss schedule')
        parser.add_argument('--patience', type=int, default=20, help='patience of early stopping')
        parser.add_argument('--min_delta', type=float, default=1e-3, help='min delta of early stopping')
        parser.add_argument('--stopping_threshold', type=float, default=0.01, help='stopping threshold of val loss for early stopping')
        return parser