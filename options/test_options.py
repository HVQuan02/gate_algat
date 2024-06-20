from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('vigat_model', nargs=1, help='Vigat trained model')
        parser.add_argument('gate_model', nargs=1, help='Gate trained model')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--save_scores', action='store_true', help='save the output scores')
        parser.add_argument('--save_path', default='scores.txt', help='output path')
        parser.add_argument('--cls_number', type=int, default=5, help='number of classifiers ')
        parser.add_argument('--t_step', nargs="+", type=int, default=[3, 5, 7, 9, 13], help='Classifier frames')
        parser.add_argument('--t_array', nargs="+", type=int, default=[1, 2, 3, 4, 5], help='e_t calculation')
        parser.add_argument('--threshold', type=float, default=0.75, help='threshold for logits to labels')
        return parser