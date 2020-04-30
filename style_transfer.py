from config import get_arguments
from SinGAN.manipulate_preprocess import *
from SinGAN.training import *
from SinGAN.imresize import imresize
import SinGAN.functions as functions

"""
This function runs SinGAN with optional styling.
The syntax for running the function is typically:
python style_transfer.py 
--input_name <name of pretrained image to use> 
--mode random_samples 
--gen_start_scale <desired modification or starting scale> 
--modification <if needed - can be of type blend or canny_color, default is None>
--ref_name <content image name - should be in the Input/Content folder generally>

"""

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='random_samples | random_samples_arbitrary_sizes', default='train')
    # for random_samples:
    parser.add_argument('--gen_start_scale', type=int, help='generation start scale', default=0)
    # for random_samples_arbitrary_sizes:
    parser.add_argument('--scale_h', type=float, help='horizontal resize factor for random samples', default=1.5)
    parser.add_argument('--scale_v', type=float, help='vertical resize factor for random samples', default=1)
    parser.add_argument('--modification', help='none | blend | canny_color', default=None)
    parser.add_argument('--ref_dir', help='input reference dir', default='Input/Content')
    parser.add_argument('--ref_name', help='reference image name', required=False)
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    in_s = functions.generate_in2coarsest(reals,1,1,opt)
    modification = opt.modification
    SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, modification, gen_start_scale=opt.gen_start_scale)




