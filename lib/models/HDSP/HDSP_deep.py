import math
import logging
from HDSP.lib.models.layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt
import torch.nn.functional as F
import torch.hub
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
_logger = logging.getLogger(__name__)
import torch
import torch.nn as nn


class ChannalAttention(nn.Module):
    def __init__(self, in_planes=768, ratio=8):

        super(ChannalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class pix_module(nn.Module):
    def __init__(self, channels=768, r=4):
        super(pix_module, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):

        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)
        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output


class TokenAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(TokenAttention, self).__init__()
        assert kernel_size in (3, 7),
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class TemporalConvolutionModel(nn.Module):
    def __init__(self, input_channels=768, output_channels=256, kernel_size=1):
        super(TemporalConvolutionModel, self).__init__()
        self.conv_layer1 = nn.Conv2d(input_channels, output_channels, kernel_size)
        self.conv_layer2 = nn.Conv2d(output_channels,input_channels, kernel_size)

        self.sigmoid = nn.Sigmoid()
    def forward(self, frame_prev, frame_next):
        temporal_diff = frame_prev - frame_next
        conv_output = self.conv_layer1(temporal_diff)
        conv_output = self.conv_layer2(conv_output)
        conv_output = self.sigmoid(conv_output)
        frame=conv_output*frame_next
        return frame

class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False,in_planes=768, ratio=8):
        super(Prompt_block, self).__init__()
        self.spatialattention0 = ChannalAttention(in_planes, ratio)
        self.tokenattention0 = TokenAttention(kernel_size=7)
        self.spatialattention1 = ChannalAttention(in_planes, ratio)
        self.tokenattention1 = TokenAttention(kernel_size=7)
        self.spatialattention2 = ChannalAttention(in_planes, ratio)
        self.tokenattention2 = TokenAttention(kernel_size=7)
        self.timeprompt = TemporalConvolutionModel()
        self.pix1=pix_module()
        self.pix2 = pix_module()
        self.fovea = Fovea(smooth=smooth)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x,i):
        B, C, W, H = x.shape
        if i==0:
            x0 = x[:, 0:768, :, :].contiguous()
            x1 = x[:, 768:1536, :, :].contiguous()
            x2 = x[:, 1536:2304, :, :].contiguous()
            x2=self.timeprompt(x2,x0)
            x0 = x0 * self.spatialattention0(x0)
            x0 = x0 * self.tokenattention0(x0)
            x1 = x1 * self.spatialattention1(x1)
            x1 = x1 * self.tokenattention1(x1)
            x2 = x2 * self.spatialattention2(x2)
            x2 = x2 * self.tokenattention2(x2)
            x0=self.fovea(x0)
            x22 = self.pix1(x0, x2)
            x11=self.pix2(x0,x1)
        else:
            x0 = x[:, 0:768, :, :].contiguous()
            x1 = x[:, 768:1536, :, :].contiguous()
            x2 = x[:, 1536:2304, :, :].contiguous()
            x0 = x0 * self.spatialattention0(x0)
            x0 = x0 * self.tokenattention0(x0)
            x1 = x1 * self.spatialattention1(x1)
            x1 = x1 * self.tokenattention1(x1)
            x2 = x2 * self.spatialattention2(x2)
            x2 = x2 * self.tokenattention2(x2)
            x0 = self.fovea(x0)
            x22 = self.pix1(x0, x2)
            x11 = self.pix2(x0, x1)


        return x11,x22

class VisionTransformerCE(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None, prompt_type=None):
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU


        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)


        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W

        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))
        self.prompt_type = prompt_type
        # various architecture
        if self.prompt_type == 'hdsp_deep':

            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'hdsp_deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=8, smooth=True))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)
            prompt_norms = []
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                    keep_ratio_search=ce_keep_ratio_i)
            )
        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):

        B, H, W = x.shape[0], x.shape[2], x.shape[3]
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # depth thermal event images
        x_dte1 = x[:, 3:6, :, :]
        x_dte2 = x[:, 6:9, :, :]
        z_dte1 = z[:, 3:6, :, :]
        z_dte2 = z[:, 6:9, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb
        z = self.patch_embed(z)
        x = self.patch_embed(x)
        z_dte1= self.patch_embed_prompt(z_dte1)
        x_dte1 = self.patch_embed_prompt(x_dte1)
        z_dte2 = self.patch_embed(z_dte2)
        x_dte2 = self.patch_embed(x_dte2)

        if self.prompt_type =='hdsp_deep':
            z_feat = token2feature(self.prompt_norms[0](z))
            x_feat = token2feature(self.prompt_norms[0](x))
            z_dte_feat1 = token2feature(self.prompt_norms[0](z_dte1))
            x_dte_feat1 = token2feature(self.prompt_norms[0](x_dte1))
            z_dte_feat2 = token2feature(self.prompt_norms[0](z_dte2))
            x_dte_feat2 = token2feature(self.prompt_norms[0](x_dte2))
            z_feat = torch.cat([z_feat, z_dte_feat1,z_dte_feat2], dim=1)
            x_feat = torch.cat([x_feat, x_dte_feat1,x_dte_feat2], dim=1)
            z_feat1,z_feat2 = self.prompt_blocks[0](z_feat,0)
            x_feat1,x_feat2 = self.prompt_blocks[0](x_feat,0)
            z_dte11 = feature2token(z_feat1)
            x_dte11 = feature2token(x_feat1)
            z_dte22 = feature2token(z_feat2)
            x_dte22 = feature2token(x_feat2)
            z_prompted1=z_dte11
            z_prompted2 = z_dte22
            x_prompted1= x_dte11
            x_prompted2 = x_dte22

            z = z + z_dte11+z_dte22
            x = x + x_dte11+x_dte22
        else:
            z = z + z_dte1+z_dte2
            x = x + x_dte1+x_dte1

        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1)

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)

        removed_indexes_s = []
        removed_flag = False
        for i, blk in enumerate(self.blocks):

            if i >= 1:
                if self.prompt_type =='hdsp_deep':
                    x_ori = x
                    # recover x to go through prompt blocks
                    lens_z_new = global_index_t.shape[1]
                    lens_x_new = global_index_s.shape[1]
                    z = x[:, :lens_z_new]
                    x = x[:, lens_z_new:]
                    if removed_indexes_s and removed_indexes_s[0] is not None:
                        removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
                        pruned_lens_x = lens_x - lens_x_new
                        pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
                        x = torch.cat([x, pad_x], dim=1)
                        index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
                        C = x.shape[-1]
                        x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
                    x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)
                    x = torch.cat([z, x], dim=1)

                    # prompt
                    x = self.prompt_norms[i - 1](x)  # todo
                    z_tokens = x[:, :lens_z, :]
                    x_tokens = x[:, lens_z:, :]
                    z_feat = token2feature(z_tokens)
                    x_feat = token2feature(x_tokens)

                    z_prompted1= self.prompt_norms[i](z_prompted1)
                    x_prompted1 = self.prompt_norms[i](x_prompted1)
                    z_prompt_feat1 = token2feature(z_prompted1)
                    x_prompt_feat1 = token2feature(x_prompted1)
                    z_prompted2 = self.prompt_norms[i](z_prompted2)
                    x_prompted2 = self.prompt_norms[i](x_prompted2)
                    z_prompt_feat2 = token2feature(z_prompted2)
                    x_prompt_feat2 = token2feature(x_prompted2)

                    z_feat = torch.cat([z_feat, z_prompt_feat1,z_prompt_feat2], dim=1)
                    x_feat = torch.cat([x_feat, x_prompt_feat1,x_prompt_feat2], dim=1)
                    z_feat1,z_feat2 = self.prompt_blocks[i](z_feat,i)
                    x_feat1,x_feat2= self.prompt_blocks[i](x_feat,i)

                    z11 = feature2token(z_feat1)
                    x11 = feature2token(x_feat1)
                    z22 = feature2token(z_feat2)
                    x22 = feature2token(x_feat2)
                    z_prompted1, x_prompted1 = z11, x11
                    z_prompted2, x_prompted2 = z22, x22
                    z=torch.cat([z11,z22],dim=1)
                    x=torch.cat([x11,x22],dim=1)
                    x = combine_tokens(z, x, mode=self.cat_mode)
                    # re-conduct CE
                    x = x_ori + candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)

            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)

        return x, aux_dict

def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_base_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model


def vit_large_patch16_224_ce_prompt(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
