import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from training.networks import Conv2dLayer, MappingNetwork, DiscriminatorBlock, DiscriminatorEpilogue
from training.networks import SynthesisNetwork as OrigSynthesisNetwork

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(OrigSynthesisNetwork):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        super().__init__(w_dim, img_resolution, img_channels, channel_base, channel_max, num_fp16_res, **block_kwargs)


    def forward(self,
                ws,
                get_feat=False,
                feat2img=False,
                feature_map=None,
                inter_image=None,
                get_inter_image=False,
                inter_image_res=64,
                **block_kwargs):
        assert (feat2img and get_inter_image) == 0, 'can not generate image with feature map and try to get intermediate image at the same time!'
        # Input the feature map, intermediate image and w latent codes and return full resolution image.
        if feat2img is True and inter_image is not None:
            img = inter_image
            x = feature_map
            feat_res = x.shape[-1]
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                ws = ws.to(torch.float32)
                w_idx = 0
                rest_resolutions = [res_rest for res_rest in self.block_resolutions if res_rest > feat_res]
                for res in rest_resolutions:
                    block = getattr(self, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv

            feats = {}
            for res, cur_ws in zip(rest_resolutions, block_ws):
                block = getattr(self, f'b{res}')
                x, img = block(x, img, cur_ws, **block_kwargs)

        elif feat2img is True and inter_image is None:
            feat_res = feature_map.shape[-1]
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in self.block_resolutions:
                    block = getattr(self, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv

            x = img = None
            feats = {}
            for res, cur_ws in zip(self.block_resolutions, block_ws):
                    block = getattr(self, f'b{res}')
                    x, img = block(x, img, cur_ws, **block_kwargs)
                    if res == feat_res:
                        x = feature_map

        # The vanilla way for G.Synthesis to forward (only using the ws).
        else:
            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in self.block_resolutions:
                    block = getattr(self, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv

            x = img = None

            feats = {}
            for res, cur_ws in zip(self.block_resolutions, block_ws):
                block = getattr(self, f'b{res}')
                x, img = block(x, img, cur_ws, **block_kwargs)
                if get_feat:
                    feats[res] = x.float()
                if get_inter_image and res == inter_image_res:
                    inter_image = img

        if get_feat is True and get_inter_image is False:
            return img, feats
        elif get_feat is False and get_inter_image is True:
            return img, inter_image
        elif get_feat is True and get_inter_image is True:
            return img, feats, inter_image
        else:
            return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        w_dim               = 512,
        decoder_res         = 64,       # A list that will be converted to int.
        glead_res_g         = 64,       # The target G resolution of GLeaD.
        decoder_out_channel = -1 ,      # The decoder final output channel num.
        decoder_conv_kernel_size = 1,   # The conv_kernel_size of the decoder.
    ):
        super().__init__()
        if isinstance(decoder_res, list):
            self.decoder_res = max(decoder_res)  # convert list to int
        else:
            self.decoder_res = decoder_res
        self.glead_res_g = glead_res_g
        self.decoder_out_channel = decoder_out_channel
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        print('channels dict: ', channels_dict)
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        self.fp16_resolution = fp16_resolution

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)

        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

        # *************************************************
        # Decoder part of D for GLeaD loss
        # *************************************************
        dec_kernel_size = int(decoder_conv_kernel_size)
        self.dec_resolutions = [2 ** i for i in range(3, int(np.log2(self.decoder_res)) + 1)]

        for res in self.dec_resolutions:
            out_channels = channels_dict[res]
            in_channels = channels_dict[res // 2]
            if res != self.dec_resolutions[0]:
                in_channels *= 2

            block = Conv2dLayer(in_channels, out_channels, kernel_size=dec_kernel_size, activation='linear', up=2)
            setattr(self, f'b{res}_dec', block)

        # Final output projection mainly for adjusting channel num.
        if decoder_out_channel == -1:
            decoder_out_channel = out_channels
        # If decoder_out_channel is 0, no projection head will be added to decoder.
        if decoder_out_channel != 0:
            # D_decoder predicts the features for reconstruction.
            if decoder_out_channel == 512:
                # FC for generate the intermediate image.
                img_proj = Conv2dLayer(channels_dict[self.decoder_res], 3, kernel_size=1, activation='linear')
                # FCs for generate the intermediate feature map and ws.
                feat_proj1 = Conv2dLayer(channels_dict[self.decoder_res], channels_dict[self.glead_res_g], kernel_size=1, activation='linear')
                feat_proj2 = Conv2dLayer(channels_dict[self.decoder_res], 512, kernel_size=1, activation='linear')
                setattr(self, 'img_proj', img_proj)
                setattr(self, 'feat_proj1', feat_proj1)
                setattr(self, 'feat_proj2', feat_proj2)
            else:
                final_proj = Conv2dLayer(out_channels, decoder_out_channel, kernel_size=dec_kernel_size, activation='linear')
                setattr(self, 'final_proj', final_proj)

    def forward(self, img, c, **block_kwargs):
        x = None
        feats = {}
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)
            feats[res // 2] = x  # Keep feature maps for U-Net decoder.

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        logits = self.b4(x, img, cmap)  # Original real/fake logits.

        # Run decoder part.
        fmaps = {}
        for idx, res in enumerate(self.dec_resolutions):
            block = getattr(self, f'b{res}_dec')
            if idx == 0:
                y = feats[res // 2]
            else:
                y = torch.cat([y, feats[res // 2]], dim=1)
            y = block(y)
            fmaps[res] = y

        final_out = None
        ws_out = None
        # If decoder_out_channel != 0, use final projection to adjust channel.
        if self.decoder_out_channel != 0:
            if self.decoder_out_channel == 512:
                img_proj = getattr(self, 'img_proj')
                feat_proj1 = getattr(self, 'feat_proj1')
                feat_proj2 = getattr(self, 'feat_proj2')
                global_pool = torch.nn.AdaptiveAvgPool2d(1)
                inter_img = img_proj(fmaps[self.decoder_res])
                inter_feat = feat_proj1(fmaps[self.decoder_res])
                ws = global_pool(feat_proj2(fmaps[self.decoder_res]))
                ws = ws.reshape(ws.shape[0], 1, ws.shape[1])
                return logits, fmaps, [inter_feat, inter_img], ws
            else:
                final_proj = getattr(self, 'final_proj')
                final_out = final_proj(fmaps[self.decoder_res])

        return logits, fmaps, final_out, ws_out

