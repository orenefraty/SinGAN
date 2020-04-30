from __future__ import print_function
import SinGAN.functions
import SinGAN.models
import argparse
import os
import random
from SinGAN.imresize import imresize
from SinGAN.imresize import imresize_to_shape
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img, feature
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from SinGAN.training import *
from config import get_arguments
import blend_modes
import SinGAN.functions as functions
from skimage.morphology import dilation,disk

def generate_gif(Gs, Zs, reals, NoiseAmp, opt, alpha=0.1, beta=0.9, start_scale=2, fps=10):
    in_s = torch.full(Zs[0].shape, 0, device=opt.device)
    images_cur = []
    count = 0

    for G, Z_opt, noise_amp, real in zip(Gs, Zs, NoiseAmp, reals):
        pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
        nzx = Z_opt.shape[2]
        nzy = Z_opt.shape[3]
        # pad_noise = 0
        # m_noise = nn.ZeroPad2d(int(pad_noise))
        m_image = nn.ZeroPad2d(int(pad_image))
        images_prev = images_cur
        images_cur = []
        if count == 0:
            z_rand = functions.generate_noise([1, nzx, nzy], device=opt.device)
            z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
            z_prev1 = 0.95 * Z_opt + 0.05 * z_rand
            z_prev2 = Z_opt
        else:
            z_prev1 = 0.95 * Z_opt + 0.05 * functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device)
            z_prev2 = Z_opt

        for i in range(0, 100, 1):
            if count == 0:
                z_rand = functions.generate_noise([1, nzx, nzy], device=opt.device)
                z_rand = z_rand.expand(1, 3, Z_opt.shape[2], Z_opt.shape[3])
                diff_curr = beta * (z_prev1 - z_prev2) + (1 - beta) * z_rand
            else:
                diff_curr = beta * (z_prev1 - z_prev2) + (1 - beta) * (
                    functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device))

            z_curr = alpha * Z_opt + (1 - alpha) * (z_prev1 + diff_curr)
            z_prev2 = z_prev1
            z_prev1 = z_curr

            if images_prev == []:
                I_prev = in_s
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                I_prev = I_prev[:, :, 0:real.shape[2], 0:real.shape[3]]
                # I_prev = functions.upsampling(I_prev,reals[count].shape[2],reals[count].shape[3])
                I_prev = m_image(I_prev)
            if count < start_scale:
                z_curr = Z_opt

            z_in = noise_amp * z_curr + I_prev
            I_curr = G(z_in.detach(), I_prev)

            if (count == len(Gs) - 1):
                I_curr = functions.denorm(I_curr).detach()
                I_curr = I_curr[0, :, :, :].cpu().numpy()
                I_curr = I_curr.transpose(1, 2, 0) * 255
                I_curr = I_curr.astype(np.uint8)

            images_cur.append(I_curr)
        count += 1
    dir2save = functions.generate_dir2save(opt)
    try:
        os.makedirs('%s/start_scale=%d' % (dir2save, start_scale))
    except OSError:
        pass
    imageio.mimsave('%s/start_scale=%d/alpha=%f_beta=%f.gif' % (dir2save, start_scale, alpha, beta), images_cur,
                    fps=fps)
    del images_cur


def SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, modification=None, in_s=None, scale_v=1, scale_h=1, n=0,
                    gen_start_scale=0, num_samples=10):
    # start_scale = here we manipulate the image
    # func stylize

    # if torch.is_tensor(in_s) == False:
    if in_s is None:
        in_s = torch.full(reals[0].shape, 0, device=opt.device)
    images_cur = []
    for G, Z_opt, noise_amp in zip(Gs, Zs, NoiseAmp):
        pad1 = ((opt.ker_size - 1) * opt.num_layer) / 2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2] - pad1 * 2) * scale_v
        nzy = (Z_opt.shape[3] - pad1 * 2) * scale_h

        images_prev = images_cur
        images_cur = []

        for i in range(0, num_samples, 1):
            if n == 0:
                z_curr = functions.generate_noise([1, nzx, nzy], device=opt.device)
                z_curr = z_curr.expand(1, 3, z_curr.shape[2], z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z, nzx, nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
                # I_prev = m(I_prev)
                # I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                # I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
            else:
                I_prev = images_prev[i]
                I_prev = imresize(I_prev, 1 / opt.scale_factor, opt)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[n].shape[2]), 0:round(scale_h * reals[n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:, :, 0:z_curr.shape[2], 0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev, z_curr.shape[2], z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            z_in = noise_amp * (z_curr) + I_prev

            dir2save = '%s/RandomSamples/%s/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], modification, gen_start_scale)

            try:
                os.makedirs(dir2save)
            except OSError:
                pass

            if n==gen_start_scale:
                plt.imsave('%s/%d_before_modification.png' % (dir2save, i), functions.convert_image_np(z_in.detach()), vmin=0,vmax=1)

            # ##################################### Image modification #################################################
            #TODO if you want the modification to happen only once, change the >= into ==
            #TODO at the moment, modification happens at every scale from the gen_start_scale and above, unless no
            #TODO modification is specificed
            #TODO The modified image is saved only at the generation scale
            #TODO when using blending, consider trying different blending options and opcity. These can be modified
            #TODO within the modify_input_to_generator function below
            if (n >= gen_start_scale) & (modification is not None):
                shape = z_in.shape
                cont_in = preprocess_content_image(opt, reals,n)
                z_in = modify_input_to_generator(z_in, cont_in, modification, opacity=1)
                assert shape == z_in.shape
                if n==gen_start_scale:
                    plt.imsave('%s/%d_after_modification.png' % (dir2save, i), functions.convert_image_np(z_in.detach()), vmin=0,vmax=1)
            # ################################## End of image modification #############################################
            I_curr = G(z_in.detach(), I_prev)

            if n == len(reals) - 1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/%s/gen_start_scale=%d' % (
                    opt.out, opt.input_name[:-4], modification, gen_start_scale)
                else:
                    #dir2save = functions.generate_dir2save(opt)
                    dir2save = '%s/RandomSamples/%s/%s/gen_start_scale=%d' % (
                    opt.out, opt.input_name[:-4], modification, gen_start_scale)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (
                        opt.mode != "paint2image"):
                    plt.imsave('%s/%d.png' % (dir2save, i), functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                    # plt.imsave('%s/%d_%d.png' % (dir2save,i,n),functions.convert_image_np(I_curr.detach()), vmin=0, vmax=1)
                    # plt.imsave('%s/in_s.png' % (dir2save), functions.convert_image_np(in_s), vmin=0,vmax=1)
            images_cur.append(I_curr)
        n += 1
    return I_curr.detach()


from skimage.color import rgb2gray,rgba2rgb
from skimage import data
import math
from skimage.transform import resize

def modify_input_to_generator(z_in, cont_in, modification, opacity=0.8):
    cuda_device = cont_in.device
    modified_image= None
    if cont_in.device.type == 'cuda':
        cont_in = cont_in.cpu()
        z_in = z_in.cpu()

    if modification == 'blend':
        z_in = np.transpose(z_in[0, :, :, :], (1, 2, 0))
        cont_in = np.transpose(cont_in[0, :, :, :], (1, 2, 0))
        z_in = np.concatenate((z_in, np.full(shape=(z_in.shape[0], z_in.shape[1], 1), fill_value=1.0)), axis=2)
        cont_in = np.concatenate((cont_in, np.full(shape=(cont_in.shape[0], cont_in.shape[1], 1), fill_value=1.0)),axis=2)
        cont_in = resize(cont_in, (z_in.shape[0], z_in.shape[1], 4), anti_aliasing=True)
        # blend - we can play with types of blends, and which image is the background and which is the foreground - only the forground (second image) is manipulated
        # modified_image = blend_modes.soft_light(cont_in, z_in, opacity)
        # modified_image = blend_modes.grain_merge(cont_in, z_in, opacity)
        modified_image = blend_modes.darken_only(cont_in, z_in, opacity)
        # modified_image = blend_modes.hard_light(cont_in, z_in, opacity)
        # modified_image = blend_modes.hard_light(z_in, cont_in, opacity)
        modified_image = rgba2rgb(modified_image)
        modified_image = np.transpose(modified_image, (2, 0, 1))
        modified_image = modified_image[np.newaxis, ...]
        modified_image = torch.from_numpy(modified_image).float().to(cuda_device)

    if modification == 'canny_color':
        cont_in_img = np.transpose(cont_in[0, :, :, :], (1, 2, 0))
        cont_in_gs = rgb2gray(cont_in_img.numpy())
        edges_bw = feature.canny(cont_in_gs)

        #edges with dilation
        edges_bw = dilation(edges_bw, selem=disk(1))

        edges = edges_bw[np.newaxis, np.newaxis, ...]
        cont_in = np.hstack((cont_in, edges))
        cont_in = np.transpose(cont_in[0, :, :, :], (1, 2, 0))
        cont_in = resize(cont_in, (z_in.shape[2] , z_in.shape[3],4), anti_aliasing=True)
        alpha = cont_in[:,:,3]!=0
        cont_in = rgba2rgb(cont_in)
        mask =np.transpose(np.tile(alpha, (3,1,1)),(1,2,0))
        z_in_rgb = np.transpose(z_in[0,:,:,:],(1,2,0))
        modified_image = np.where(mask,cont_in,z_in_rgb)
        modified_image = np.transpose(modified_image, (2, 0, 1))
        modified_image = modified_image[np.newaxis, ...]
        modified_image = torch.from_numpy(modified_image).float().to(cuda_device)
    return modified_image


def preprocess_content_image(opt, reals,scale):
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    ref = functions.read_image_dir('%s/%s' % (opt.ref_dir, opt.ref_name), opt)
    if ref.shape[3] != real.shape[3]:
        ref = imresize_to_shape(ref, [real.shape[2], real.shape[3]], opt)
        ref = ref[:, :, :real.shape[2], :real.shape[3]]

    N = len(reals) - 1
    n = scale
    in_s = imresize(ref, pow(opt.scale_factor, (N - n + 1)), opt)
    in_s = in_s[:, :, :reals[n - 1].shape[2], :reals[n - 1].shape[3]]
    in_s = imresize(in_s, 1 / opt.scale_factor, opt)
    in_s = in_s[:, :, :reals[n].shape[2], :reals[n].shape[3]]

    return in_s
