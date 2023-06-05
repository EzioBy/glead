# Under NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator Augmentation (ADA).

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.loss import Loss
from training.lpips.lpips import LPIPS


class StyleGAN2GLeaDLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 glead_res_d=[64], glead_res_g=64,
                 reg_target_real='gfeat', reg_weight_real=10.0, reg_loss_real=None,
                 reg_target_fake='gfeat', reg_weight_fake=10.0, reg_loss_fake=None):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.glead_res_d = glead_res_d
        self.glead_res_d_max = max(glead_res_d)
        self.glead_res_g = glead_res_g
        # Regularization target - e.g., image reconstruction (GLeaD) or feature matching (GGDR).
        self.reg_target_real = reg_target_real
        self.reg_target_fake = reg_target_fake
        # Regularization loss type - e.g., MSE or LPIPS.
        self.reg_loss_real = reg_loss_real
        self.reg_loss_fake = reg_loss_fake
        # Regularization loss weight.
        self.reg_weight_real = reg_weight_real
        self.reg_weight_fake = reg_weight_fake
        print('-' * 20)
        print('The GLeaD related settings: |')
        print(f'glead_res_D: {self.glead_res_d}   |')
        print(f'glead_res_G: {self.glead_res_g}   |')
        print(f'reg_real: {self.reg_target_real} - {self.reg_loss_real} - {self.reg_weight_real} |')
        print(f'reg_fake: {self.reg_target_fake} - {self.reg_loss_fake} - {self.reg_weight_fake} |')
        print('-' * 20)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        # Establish LPIPS network first if needed.
        if 'lpips' in str(self.reg_loss_real).lower() or 'lpips' in str(self.reg_loss_fake).lower() or self.reg_target_real == 'gfwimage' or self.reg_target_fake == 'gfwimage':
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()

    def run_G(self, z, c, ws=None, get_inter_image=False, inter_image_res=64, sync=True):
        with misc.ddp_sync(self.G_mapping, sync):
            if ws is None:
                ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            if get_inter_image is False:
                img, output_feat = self.G_synthesis(ws, get_feat=True, get_inter_image=get_inter_image)
                return img, ws, output_feat
            else:
                img, output_feat, inter_img = self.G_synthesis(ws, get_feat=True, get_inter_image=get_inter_image, inter_image_res=inter_image_res)
                return img, ws, output_feat, inter_img


    def run_aug_if_needed(self, img, gfeats):
        """
        Augment image and feature map consistently
        """
        if self.augment_pipe is not None:
            aug_img, gfeats = self.augment_pipe(img, gfeats)
        else:
            aug_img = img
        return aug_img, gfeats

    def run_D(self, img, c, gfeats=None, sync=None):
        aug_img, gfeats = self.run_aug_if_needed(img, gfeats)
        with misc.ddp_sync(self.D, sync):
            logits, decoder_out_list, final_out, ws_out = self.D(aug_img, c)
        return logits, decoder_out_list, final_out, ws_out, aug_img, gfeats

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # The vanilla GAN loss for G.
                gen_img, _gen_ws, _gen_feat = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))
                gen_logits, _recon_gen_fmaps, _, _, _, _ = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gadv = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss_adv', loss_Gadv)
                loss_Gmain = loss_Gadv
                training_stats.report('Loss/G/loss_all', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, gen_fmaps = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                loss_gen_reg = 0.0
                gen_img, _gen_ws, gen_fmaps = self.run_G(gen_z, gen_c, sync=sync)

                aug_gen_logits, aug_decoder_gen_fmaps, decoder_final_out, decoder_ws_out, aug_gen_img, aug_fmaps = \
                    self.run_D(gen_img, gen_c, gen_fmaps, sync=sync)
                decoder_inter_image = None
                if isinstance(decoder_final_out, list):
                    if len(decoder_final_out) == 2:
                        decoder_final_out, decoder_inter_image = decoder_final_out[0], decoder_final_out[1]
                        if decoder_inter_image is not None:
                            loss_gen_reg += decoder_inter_image[:, 0, 0, 0] * 0
                        if decoder_ws_out is not None:
                            loss_gen_reg += decoder_ws_out[:, 0, 0] * 0
                    else:
                        decoder_final_out = decoder_final_out[self.glead_res_d_max]

                loss_gan_gen = torch.nn.functional.softplus(aug_gen_logits) + \
                    aug_decoder_gen_fmaps[max(aug_decoder_gen_fmaps.keys())][:, 0, 0, 0] * 0

                # Set reg phase.
                if self.reg_target_fake == 'None':
                    loss_gen_reg += aug_decoder_gen_fmaps[max(aug_decoder_gen_fmaps.keys())][:, 0, 0, 0] * 0
                    if decoder_final_out is not None:
                        loss_gen_reg += decoder_final_out[:, 0, 0, 0] * 0
                else:
                    # Gfeat mode: only match the G features of fake images (like GGDR).
                    if self.reg_target_fake == 'gfeat':
                        if self.reg_loss_fake is None:
                            self.reg_loss_fake = 'cos'
                        if self.reg_target_real in ['gfeat', 'None']:  # no extra head exists, just use D_gen_fmaps as source.
                            source = aug_decoder_gen_fmaps[self.glead_res_d_max]
                        elif self.reg_target_real == 'gfwimage':  # use decoder_final_out as source when head exists.
                            source = decoder_final_out
                            # resize D_decoder output if needed.
                            if decoder_final_out.shape[-1] != self.glead_res_g:
                                source = F.interpolate(source,
                                                       (self.glead_res_g, self.glead_res_g),
                                                       mode='bilinear',
                                                       align_corners=False)
                        target = aug_fmaps[self.glead_res_g]

                    # Gfwimage: reconstruct fake images with D and G.
                    elif self.reg_target_fake == 'gfwimage':
                        if self.reg_loss_fake is None:
                            self.reg_loss_fake = 'lpips'
                        target = gen_img
                        if decoder_final_out.shape[-1] != self.glead_res_g:
                            decoder_final_out = F.interpolate(decoder_final_out,
                                                              (self.glead_res_g, self.glead_res_g),
                                                               mode='bilinear',
                                                               align_corners=False)
                            decoder_inter_image = F.interpolate(decoder_inter_image,
                                                                (self.glead_res_g, self.glead_res_g),
                                                                 mode='bilinear',
                                                                 align_corners=False)
                        w_amount = int(np.log2(gen_img.shape[-1] // self.glead_res_g)) * 2 + 1
                        decoder_ws_out = decoder_ws_out.repeat(1, w_amount, 1)

                        source = self.G_synthesis(get_feat=False,
                                                  feat2img=True,
                                                  ws=decoder_ws_out,
                                                  feature_map=decoder_final_out,
                                                  inter_image=decoder_inter_image)

                    loss_gen_reg += self.get_glead_reg_single_res(self.reg_loss_fake, self.reg_weight_fake, source, target)

                loss_Dmain = loss_gan_gen + loss_gen_reg

                training_stats.report('Loss/D/loss_gan_gen', loss_gan_gen)
                training_stats.report(f'Loss/D/loss_fake_reg_{self.reg_loss_fake}', loss_gen_reg)

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dmain.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                loss_real_reg = 0.0
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, aug_decoder_real_fmaps, decoder_final_out, decoder_ws_out, _, _ \
                    = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                decoder_inter_image = None
                if isinstance(decoder_final_out, list):
                    if len(decoder_final_out) == 2:
                        decoder_final_out, decoder_inter_image = decoder_final_out[0], decoder_final_out[1]
                        if self.reg_target_real != 'gfwimage':
                            if decoder_inter_image is not None:
                                loss_real_reg += decoder_inter_image[:, 0, 0, 0] * 0
                            if decoder_ws_out is not None:
                                loss_real_reg += decoder_ws_out[:, 0, 0] * 0

                    else:
                        decoder_final_out = decoder_final_out[self.glead_res_d_max]

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report(f'Loss/D/loss', loss_Dreal + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

                # Handle unused vars.
                if self.reg_target_real == 'None':
                    loss_real_reg += aug_decoder_real_fmaps[max(aug_decoder_real_fmaps.keys())][:, 0, 0, 0] * 0
                    if decoder_final_out is not None:
                        loss_real_reg += decoder_final_out[:, 0, 0, 0] * 0
                else:
                    # Reconstruct real images with D and G.
                    if self.reg_target_real == 'gfwimage':
                        if self.reg_loss_real is None:
                            self.reg_loss_real = 'lpips'
                        target = real_img_tmp
                        if decoder_final_out.shape[-1] != self.glead_res_g:
                            decoder_final_out = F.interpolate(decoder_final_out,
                                                              (self.glead_res_g, self.glead_res_g),
                                                              mode='bilinear',
                                                              align_corners=False)
                            decoder_inter_image = F.interpolate(decoder_inter_image,
                                                                (self.glead_res_g, self.glead_res_g),
                                                                mode='bilinear',
                                                                align_corners=False)

                        w_amount = int(np.log2(real_img_tmp.shape[-1] // self.glead_res_g)) * 2 + 1
                        decoder_ws_out = decoder_ws_out.repeat(1, w_amount, 1)
                        source = self.G_synthesis(get_feat=False,
                                                  feat2img=True,
                                                  ws=decoder_ws_out,
                                                  feature_map=decoder_final_out,
                                                  inter_image=decoder_inter_image)

                    # Calculate the target reg loss.
                    loss_real_reg += self.get_glead_reg_single_res(self.reg_loss_real, self.reg_weight_real, source, target)
                    training_stats.report(f'Loss/D/loss_real_reg_{self.reg_loss_real}', loss_real_reg)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_real_reg + loss_Dreal + loss_Dr1 + real_logits * 0).mean().mul(gain).backward()

    def cosine_distance(self, x, y):
        return 1. - F.cosine_similarity(x, y).mean()

    # GLeaD reg for MUTIPLE resolutions (we do NOT use this in the current version).
    def get_glead_reg(self, reg_weight, glead_resolutions, source, target):
        loss_gen_recon = 0
        for res in glead_resolutions:
            loss_gen_recon += reg_weight * self.cosine_distance(source[res], target[res]) / len(glead_resolutions)

        return loss_gen_recon

    # GLeaD reg for single resolution.
    def get_glead_reg_single_res(self, loss_type, reg_weight, source, target):
        if loss_type.lower() == 'mse':
            loss_gen_recon = reg_weight * F.mse_loss(source, target)
        elif loss_type.lower() == 'lpips':
            assert self.lpips_loss, 'LPIPS net has not been established!'
            assert source.shape[-1] == target.shape[-1], f'source.shape:{source.shape} and target.shape:{target.shape} must be equal!'
            loss_gen_recon = reg_weight * self.lpips_loss(source, target)
        elif loss_type.lower() == 'cos':
            loss_gen_recon = reg_weight * self.cosine_distance(source, target)
        else:
            print(f'Loss type {loss_type} is not supported, thus None is returned!!!')
            loss_gen_recon = None
        return loss_gen_recon
