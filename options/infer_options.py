from .base_options import BaseOptions

class InferOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('vigat_model', nargs=1, help='Vigat trained model')
        parser.add_argument('gate_model', nargs=1, help='Gate trained model')
        parser.add_argument('--album_path', type=str, default='/kaggle/working/gate_vigat/albums/Graduation/0_92024390@N00')
        parser.add_argument('--path_output', type=str, default='/kaggle/working/outputs')
        return parser