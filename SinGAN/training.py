import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

def train(opt,Gs,Zs,reals1,reals2,NoiseAmp):
    real1_, real2_ = functions.read_image(opt)
    in_s = 0
    scale_num = 0
    real1 = imresize(real1_,opt.scale1,opt)
    real2 = imresize(real2_, opt.scale1, opt)
    reals1 = functions.creat_reals_pyramid(real1,reals1,opt)
    reals2 = functions.creat_reals_pyramid(real2, reals2, opt)
    nfc_prev = 0
    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        plt.imsave('%s/real1_scale.png' %  (opt.outf), functions.convert_image_np(reals1[scale_num]), vmin=0, vmax=1)
        plt.imsave('%s/real2_scale.png' % (opt.outf), functions.convert_image_np(reals2[scale_num]), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals1,reals2,Gs,Zs,in_s,NoiseAmp,opt)

        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr,False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals1, '%s/reals1.pth' % (opt.out_))
        torch.save(reals2, '%s/reals2.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return



def train_single_scale(netD,netG,reals1,reals2,Gs,Zs,in_s,NoiseAmp,opt,centers=None):

    # TODO: check that the size of the first and second image is the same at this stage
    real = reals1[len(Gs)] #this line was not adapted for two images because it is only used for size manipulations
    opt.nzx = real.shape[2]
    opt.nzy = real.shape[3]
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    if opt.mode == 'animation_train':
        opt.nzx = real.shape[2]+(opt.ker_size-1)*(opt.num_layer)
        opt.nzy = real.shape[3]+(opt.ker_size-1)*(opt.num_layer)
        pad_noise = 0
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    real1 = reals1[len(Gs)]
    real2 = reals1[len(Gs)]

    for epoch in range(opt.niter):
        for train_image in range(2):
            if train_image == 0:
                real = real1
                reals = reals1
            elif train_image == 1:
                real = real2
                reals = reals2

            if (Gs == []) & (opt.mode != 'SR_train'):
                z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
                z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
                noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
                noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
            else:
                noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
                noise_ = m_noise(noise_)

            ############################
            # (1) Update D network: maximize D(x) + D(G(z))
            ###########################
            for j in range(opt.Dsteps):
                # train with real
                netD.zero_grad()

                output = netD(real,train_image).to(opt.device)
                errD_real = -output.mean()#-a
                errD_real.backward(retain_graph=True)
                D_x = -errD_real.item()

                # train with fake
                if (j==0) & (epoch == 0):
                    if (Gs == []) & (opt.mode != 'SR_train'):
                        prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                        in_s = prev
                        prev = m_image(prev)
                        z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                        z_prev = m_noise(z_prev)
                        opt.noise_amp = 1
                    elif opt.mode == 'SR_train':
                        z_prev = in_s
                        criterion = nn.MSELoss()
                        RMSE = torch.sqrt(criterion(real, z_prev))
                        opt.noise_amp = opt.noise_amp_init * RMSE
                        z_prev = m_image(z_prev)
                        prev = z_prev
                    else:
                        prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt, train_image)
                        prev = m_image(prev)
                        z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt, train_image)
                        criterion = nn.MSELoss()
                        RMSE = torch.sqrt(criterion(real, z_prev))
                        opt.noise_amp = opt.noise_amp_init*RMSE
                        z_prev = m_image(z_prev)
                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt, train_image)
                    prev = m_image(prev)

                if opt.mode == 'paint_train':
                    prev = functions.quant2centers(prev,centers)
                    plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

                if (Gs == []) & (opt.mode != 'SR_train'):
                    noise = noise_
                else:
                    noise = opt.noise_amp*noise_+prev

                fake = netG(noise.detach(),prev,train_image)
                output = netD(fake.detach(),train_image)
                errD_fake = output.mean()
                errD_fake.backward(retain_graph=True)
                D_G_z = output.mean().item()

                gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device, train_image)
                gradient_penalty.backward()

                errD = errD_real + errD_fake + gradient_penalty
                optimizerD.step()

            errD2plot.append(errD.detach())

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            for j in range(opt.Gsteps):
                netG.zero_grad()
                output = netD(fake, train_image)
                errG = -output.mean()
                errG.backward(retain_graph=True)
                if alpha!=0:
                    loss = nn.MSELoss()
                    if opt.mode == 'paint_train':
                        z_prev = functions.quant2centers(z_prev, centers)
                        plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                    Z_opt = opt.noise_amp*z_opt+z_prev
                    rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev,train_image),real)
                    rec_loss.backward(retain_graph=True)
                    rec_loss = rec_loss.detach()
                else:
                    Z_opt = z_opt
                    rec_loss = 0

                optimizerG.step()

            errG2plot.append(errG.detach()+rec_loss)
            D_real2plot.append(D_x)
            D_fake2plot.append(D_G_z)
            z_opt2plot.append(rec_loss)

            if epoch % 25 == 0 or epoch == (opt.niter-1):
                print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

            if epoch % 500 == 0 or epoch == (opt.niter-1):
                plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
                plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev, train_image).detach()), vmin=0, vmax=1)

                torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

            schedulerD.step()
            schedulerG.step()

    functions.save_networks(netG,netD,z_opt,opt)
    return z_opt,in_s,netG    

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt, style_idx):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z, style_idx)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z, style_idx)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
    return G_z

def train_paint(opt,Gs,Zs,reals,NoiseAmp,centers,paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        if scale_num!=paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_,scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                    pass

            plt.imsave('%s/in_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr,G_curr = init_models(opt)

            z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals[:scale_num+1],Gs[:scale_num],Zs[:scale_num],in_s,NoiseAmp[:scale_num],opt,centers=centers)

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num+=1
            nfc_prev = opt.nfc
        del D_curr,G_curr
    return


def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
