from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name1', help='input image 1 name', required=True)
    parser.add_argument('--input_name2', help='input image 2 name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals1 = []
    reals2 = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real1, real2 = functions.read_image(opt)
        functions.adjust_scales2image(real1, opt)
        functions.adjust_scales2image(real2, opt)
        # TODO: do we need two reals?
        train(opt, Gs, Zs, reals1, reals2, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals1,NoiseAmp,opt,0)
        SinGAN_generate(Gs,Zs,reals2,NoiseAmp,opt,1)

