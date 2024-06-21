from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('vigat_model', nargs=1, help='Vigat trained model')
        parser.add_argument('gate_model', nargs=1, help='Gate trained model')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--save_scores', action='store_true', help='save the output scores')
        parser.add_argument('--save_path', default='scores.txt', help='output path')
        return parser