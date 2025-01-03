import copy
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.builder import MODELS
from models.tcl.gumbel import gumbel_sigmoid
from models.tcl.modules import FeatureEncoder

from utils import get_logger
import us
from models.tcl.aspp import ASPP
import shared


@MODELS.register_module()
class Sim2Mask(nn.Module):
    def __init__(self, init_w=1.0, init_b=0.0, gumbel_tau=1.0, learnable=True):
        super().__init__()
        self.init_w = init_w
        self.init_b = init_b
        self.gumbel_tau = gumbel_tau
        # self.learnable = learnable
        self.learnable = False

        assert not ((init_w is None) ^ (init_b is None))
        if learnable:
            self.w = nn.Parameter(torch.full([], float(init_w)))
            self.b = nn.Parameter(torch.full([], float(init_b)))
        else:
            self.w = init_w
            self.b = init_b

    def forward(self, x, deterministic=False):
        logits = x * self.w + self.b
        # logits = logits.float()

        soft_mask = torch.sigmoid(logits)
        if deterministic:
            hard_mask = soft_mask.gt(0.5).type(logits.dtype)
        else:
            hard_mask = gumbel_sigmoid(logits, hard=True, tau=self.gumbel_tau)

        return hard_mask, soft_mask

    def extra_repr(self):
        return f'init_w={self.init_w}, init_b={self.init_b}, learnable={self.learnable}, gumbel_tau={self.gumbel_tau}'


class MaskerBackbone(nn.Module):
    """Masker image encoder backbone.
    """
    def __init__(self, clip_visual, freeze_idx):
        super().__init__()
        self.transformer = copy.deepcopy(clip_visual.transformer)
        self.transformer.resblocks = self.transformer.resblocks[freeze_idx:]

        for block in self.transformer.resblocks:
            if hasattr(block, "hook_handler"):
                block.hook_handler.remove()

        self.ln_post = copy.deepcopy(clip_visual.ln_post)
        self.proj = copy.deepcopy(clip_visual.proj)

        self.layers = len(self.transformer.resblocks)
        self.patch_size = clip_visual.patch_size

        self.output_dim = clip_visual.output_dim if self.proj is not None else clip_visual.width

    def forward(self, x, spatial=True, ignore_last_attn=True):
        if self.layers:
            x = self.transformer(x, ignore_last_attn=ignore_last_attn)

        x = x.permute(1, 0, 2)  # LND -> NLD

        if spatial:
            x = self.ln_post(x)
        else:
            x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class MaskerImageFeatureEncoder(FeatureEncoder):
    def __init__(self, backbone: nn.Module, decoder: nn.Module, ignore_last_attn: bool = True):
        super().__init__()
        self.ignore_last_attn = ignore_last_attn
        self.patch_size = backbone.patch_size
        self.backbone = backbone
        self.decoder = decoder
        self.aspp = ASPP(768, 768)
        '''self.conv = nn.Sequential(
            nn.Conv2d(1280, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )'''

        for resblock in self.backbone.transformer.resblocks:
            resblock.hook_handler = resblock.register_forward_hook(self.hook)

    def _encode(self, image, image_feat):
        H, W = image.shape[-2:]
        h = H // self.patch_size
        w = W // self.patch_size

        x = self.backbone(image_feat, spatial=True, ignore_last_attn=self.ignore_last_attn)  # BLC
        x = rearrange(x[:, 1:], "B (H W) C -> B C H W", H=h, W=w)
        x = self.aspp(x)
        x = self.decoder(x)
        '''x_aspp = F.interpolate(x_aspp, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.concat([x, x_aspp], dim=1)
        x = self.conv(x)'''

        return x


@MODELS.register_module()
class Masker(nn.Module):
    def __init__(self, backbone, decoder, image_proj, sim2mask, ignore_last_attn, **kwargs):
        super().__init__()
        self.ignore_last_attn = ignore_last_attn

        decoder["C"] = backbone.output_dim
        decoder = MODELS.build(decoder)
        decoder = nn.Sequential(OrderedDict([
            ("decoder", decoder),
            ("image_proj", image_proj)
        ]))

        self.image_encoder = MaskerImageFeatureEncoder(backbone, decoder, ignore_last_attn=ignore_last_attn)

        self.w_m = nn.Parameter(torch.full([], float(3.0)))
        self.b_m = nn.Parameter(torch.full([], float(0.0)))
        self.sim2mask = Sim2Mask(**sim2mask)
    def forward(self, image_emb, text_emb, matrix, deterministic=False):
    # def forward(self, image_emb, text_emb, deterministic=False):
        B = image_emb.size(0)

        H, W = image_emb.shape[2:]
        D = dist.get_world_size()

        # matrix = matrix * 3
        matrix = matrix * self.w_m + self.b_m
        # simmap [B, B*D, H, W] where D is #devices
        all_text_emb_norm = us.gather_cat(text_emb, grad=True, contiguous_grad=True)
        simmap = torch.einsum("bchw,nc->bnhw", image_emb, all_text_emb_norm)  # shape [128, 128, 56, 56]
        matrix = torch.cat((matrix, matrix), dim=0).unsqueeze(2).unsqueeze(3)
        simmap = simmap * matrix
        mask, soft_mask = self.sim2mask(simmap, deterministic=deterministic)


        # mask [B, B*D, H, W] where D is #devices
        # positive global label
        pos_indices = torch.arange(B, dtype=torch.long, device=image_emb.device) + B * dist.get_rank()
        pos_mask = mask[torch.arange(B), pos_indices].unsqueeze(1)  # [B, 1, H, W]

        offdiag = torch.ones(B, B*D, dtype=torch.bool, device=mask.device)
        offdiag[torch.arange(B), pos_indices] = False

        soft_pos_mask = soft_mask[torch.arange(B), pos_indices].unsqueeze(1)
        soft_neg_mask = soft_mask.masked_select(offdiag[..., None, None]).view(B, B*D-1, H, W)

        masks = {
            "pos": pos_mask,  # [B, 1, H, W]
            "soft_pos": soft_pos_mask,
            "soft_neg": soft_neg_mask,
            "soft_all": soft_mask,  # [B, N, H, W]
        }

        return masks

    @torch.no_grad()
    def forward_seg(self, image, image_feat, text_emb, img_fea_encoder, deterministic=True, hard=False):
        """Make mask by 1:N matching

        Args:
            image [B, 3, H, W]
            image_feat [L, B, C]: CLIP features
            text_emb [N, C]
            deterministic (bool): deterministic inference flag for gumbel noise
            hard (bool): decide hard or soft returning segmentation mask.
                Note that soft mask is required for proper evaluation

        Return:
            mask [B, N, H', W'] (H' and W' are downsampled H/W)
        """
        image_emb = self.image_encoder(image, image_feat)  # [BCHW]

        image_emb = us.normalize(image_emb, dim=1)  # BCHW
        text_emb = us.normalize(text_emb, dim=-1)  # NC
        
        simmap = torch.einsum("b c h w, n c -> b n h w", image_emb, text_emb)
        image_emb_fea = img_fea_encoder(image_emb)
        pooling = nn.AdaptiveAvgPool2d(1)
        image_emb_fea = pooling(image_emb_fea).squeeze(-2).squeeze(-1)
        similarity = torch.einsum("ik, jk -> ij", image_emb_fea, text_emb).unsqueeze(-1).unsqueeze(-1)
        similarity = torch.sigmoid(similarity)
        # similarity = similarity * 3
        similarity = similarity * self.w_m + self.b_m
        simmap = simmap * similarity
        hard_mask, soft_mask = self.sim2mask(simmap, deterministic=deterministic)
        mask = hard_mask if hard else soft_mask

        return mask, simmap
