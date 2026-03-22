"""
The code in this file has been copied from MiDiffusion:
https://github.com/MIT-SPARK/MiDiffusion/blob/main/midiffusion/networks/denoising_net/continuous_transformer.py
and the related files.

We didn't make any changes to the code apart from applying our formatters for the code
to pass the CI checks. We avoided cleaning up the code or making other changes to prevent
us from accidentally deviating from the original model.
"""

import math

import torch
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from torch import Tensor, nn

LAYER_NROM_EPS = 1e-5  # pytorch's default: 1e-5

import torch

from torch import nn
from torchvision import models
from typing import Union

class BaseFeatureExtractor(nn.Module):
    """Hold some common functions among all feature extractor networks"""

    @property
    def feature_size(self):
        return self._feature_size

    def forward(self, data):
        raise NotImplementedError



class PointNet_Point(BaseFeatureExtractor):
    """The 'pointnet_simple' encoder in train script"""

    """https://github.com/QiuhongAnnaWei/LEGO-Net/blob/main/model/floorplan_encoder.py"""

    def __init__(
        self,
        activation=torch.nn.LeakyReLU(),
        nfpbpn=256,
        feat_units=[4, 64, 64, 512, 511],
    ):
        """feat_units[0] should be 4 (x, y, nx, ny for fpbpn) (in_dim)
        feat_units[-1] should be transformer_input_d-1 (out_dim)
        maxpool happens before the last linear layer
        """
        super().__init__()
        self.activation = activation
        self._feature_size = feat_units[-1]

        layers = []
        for i in range(1, len(feat_units)):
            layers.append(torch.nn.Linear(feat_units[i - 1], feat_units[i]))
        self.layers = torch.nn.ModuleList(layers)

        self.fp_maxpool = torch.nn.MaxPool1d(nfpbpn)  # nfpbpn -> 1

    def forward(self, fpbpn):
        """fpbpn: [batch_size, nfpbp=250, 4], floor_plan_boundary_points_normals"""

        B = fpbpn.shape[0]

        for i in range(len(self.layers) - 1):  # first 3 layers
            fpbpn = self.activation(self.layers[i](fpbpn))  # [B, nfpbp, feat_units[-2]]

        fpbpn = fpbpn.permute((0, 2, 1))  # [B, feat_units[-2], nfpbp]
        scene_fp_feat = self.fp_maxpool(fpbpn).reshape(
            B, -1
        )  # [B, feat_units[-2], 1] -> [B, feat_units[-2]]

        return self.layers[-1](scene_fp_feat)  # [B, feat_units[-1]]


class PointNet_Line(nn.Module):
    """The 'pointnet' encoder in train script.
    Has 3 sections: point processing, line processing, floor plan feature processing"""

    def __init__(
        self,
        activation=torch.nn.LeakyReLU(),
        maxnfpoc=25,
        corner_feat_units=[2, 64, 128],
        line_feat_units=[256, 512, 1024],
        fp_units=[1024, 511],
    ):
        """corner_feat_units[0] should be pos_dim (in_dim)
        line_feat_units[0] should be corner_feat_units[-1]*2
        fp_units[0] should be line_feat_units[-1]
        fp_units[-1] should be transformer_input_d-1 (out_dim)
        """
        super().__init__()
        self.activation = activation  # torch.nn.LeakyReLU()
        self._feature_size = fp_units[-1]

        corner_feat = []
        for i in range(1, len(corner_feat_units)):  # 2 layers
            corner_feat.append(
                torch.nn.Linear(corner_feat_units[i - 1], corner_feat_units[i])
            )
        self.corner_feat = torch.nn.ModuleList(corner_feat)

        line_feat = []
        for i in range(1, len(line_feat_units)):  # 2 layers
            line_feat.append(
                torch.nn.Linear(line_feat_units[i - 1], line_feat_units[i])
            )
        self.line_feat = torch.nn.ModuleList(line_feat)
        self.fp_maxpool = torch.nn.MaxPool1d(maxnfpoc)  # maxnfpoc -> 1

        # want floor plan to appear as an object token
        fp_feat = []
        for i in range(1, len(fp_units)):  # 1 layer
            fp_feat.append(torch.nn.Linear(fp_units[i - 1], fp_units[i]))
        self.fp_feat = torch.nn.ModuleList(fp_feat)

    def forward(self, fpoc, nfpc):
        """fpoc        : [batch_size, maxnfpoc, pos=2], with padded 0 beyond the num of floor plan ordered corners for each scene
        nfpc        : [batch_size], num of floor plan ordered corners for each scene
        """
        device = fpoc.device

        # each point -> MLP -> each point has point features
        for i in range(len(self.corner_feat) - 1):
            fpoc = self.activation(self.corner_feat[i](fpoc))
        fpoc = self.corner_feat[-1](fpoc)  # [B, nfpv, corner_feat_units[-1]=128]

        # concatenate 2 point features for each line pair -> MLP -> maxpool
        B, maxnfpoc, cornerpt_feat_d = (
            fpoc.shape[0],
            fpoc.shape[1],
            fpoc.shape[2],
        )  # maxnfpoc = maxnfpoc
        line_pairpt_input = torch.zeros(B, maxnfpoc, cornerpt_feat_d * 2).to(device)
        for s_i in range(B):
            for l_i in range(
                nfpc[s_i]
            ):  # otherwise padded with 0; clockwise ordering of lines
                line_pairpt_input[s_i, l_i, :cornerpt_feat_d] = fpoc[
                    s_i, l_i, :
                ]  # index slicing does not make copy
                line_pairpt_input[s_i, l_i, cornerpt_feat_d:] = fpoc[
                    s_i, (l_i + 1) % nfpc[s_i], :
                ]
        line_pairpt_input = line_pairpt_input.to(device)

        for i in range(len(self.line_feat) - 1):
            line_pairpt_input = self.activation(self.line_feat[i](line_pairpt_input))
        line_pairpt_input = self.line_feat[-1](
            line_pairpt_input
        )  # [B, nfpv, line_feat_units[-1]=1024]

        line_pairpt_input_padded = torch.zeros(line_pairpt_input.shape).to(
            device
        )  # [B, nfpv, 1024]
        for s_i in range(B):
            line_pairpt_input_padded[s_i, : nfpc[s_i], :] = line_pairpt_input[
                s_i, : nfpc[s_i], :
            ]
            line_pairpt_input_padded[s_i, nfpc[s_i] :, :] = line_pairpt_input[
                s_i, 0, :
            ]  # duplicate first feat to not impact maxpool
        line_pairpt_input_padded = line_pairpt_input_padded.to(device)

        line_pairpt_input_padded = line_pairpt_input_padded.permute(
            (0, 2, 1)
        )  # [B, 1024, nfpv]
        scene_fpoc = self.fp_maxpool(line_pairpt_input_padded).reshape(
            B, -1
        )  # [B, 1024, 1] -> [B, 1024]

        # One floor plan feat per scene -> MLP -> input to transformer as last obj token
        for i in range(len(self.fp_feat) - 1):
            scene_fpoc = self.activation(self.fp_feat[i](scene_fpoc))
        scene_fpoc = torch.unsqueeze(
            self.fp_feat[-1](scene_fpoc), dim=1
        )  # [B, None->1, transformer_input_d-1]

        return scene_fpoc


def get_feature_extractor(name, **kwargs):
    """Based on the name return the appropriate feature extractor."""
    if name == "resnet18":
        raise NotImplementedError
        # return ResNet18(**kwargs)
    elif name == "pointnet_simple":
        return PointNet_Point(**kwargs)
    elif name == "pointnet":
        return PointNet_Line(**kwargs)
    else:
        raise NotImplemented


def load_floor_encoder_from_config(feature_encoder: str = "pointnet_simple"):
    kwargs = {"feat_units": [4, 64, 64, 512, 64]}
    return (
        PointNet_Point(**kwargs),
        kwargs["feat_units"][-1],
    )

class SinusoidalPosEmb(nn.Module):
    """https://github.com/microsoft/VQ-Diffusion/blob/main/image_synthesis/modeling/transformers/transformer_utils.py"""

    def __init__(self, dim: int, num_steps: int = 4000, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        if num_steps != rescale_steps:
            self.num_steps = float(num_steps)
            self.rescale_steps = float(rescale_steps)
            self.input_scaling = True
        else:
            self.input_scaling = False

    def forward(self, x: Tensor):
        # (B) -> (B, self.dim)
        if self.input_scaling:
            x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(1.702 * x)


class _AdaNorm(nn.Module):
    """Base normalization layer that incorporate timestep embeddings"""

    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        assert n_embd % 2 == 0
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(n_embd, num_steps=max_timestep)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    """Norm layer modified to incorporate timestep embeddings"""

    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(
            n_embd, eps=LAYER_NROM_EPS, elementwise_affine=False
        )

    def forward(self, x: Tensor, timestep: Tensor):
        # (B, N, n_embd),(B,) -> (B, N, n_embd)
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)  # B, 1, 2*n_embd
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class AdaInsNorm(_AdaNorm):
    """Base instance normalization layer that incorporate timestep embeddings"""

    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adainsnorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.instancenorm = nn.InstanceNorm1d(n_embd, eps=LAYER_NROM_EPS)

    def forward(self, x: Tensor, timestep: Tensor):
        # (B, N, n_embd),(B,) -> (B, N, n_embd)
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)  # B, 1, 2*n_embd
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = (
            self.instancenorm(x.transpose(-1, -2)).transpose(-1, -2) * (1 + scale)
            + shift
        )
        return x


class SelfAttention(nn.Module):
    """Multi-head self attention with explicit q/k/v/out projections (LoRA-friendly)."""

    def __init__(self, n_embd, n_head, dropout=0.1, batch_first=True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        # Separate Linear layers so PEFT can target them by name
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd)
        self.v_proj = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        import torch.nn.functional as F
        B, N, C = x.shape
        H, D = self.n_head, self.head_dim
        Q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        K = self.k_proj(x).view(B, N, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, D).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * (D ** -0.5)       # (B, H, N, N)
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross attention with explicit q/k/v/out projections (LoRA-friendly)."""

    def __init__(self, n_embd, n_head, dropout=0.1, batch_first=True, kv_embd=None):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        kv_embd = kv_embd if kv_embd is not None else n_embd
        # Separate Linear layers so PEFT can target them by name
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(kv_embd, n_embd)
        self.v_proj = nn.Linear(kv_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, q, kv, attn_mask=None, key_padding_mask=None):
        import torch.nn.functional as F
        B, Nq, _ = q.shape
        Nk = kv.shape[1]
        H, D = self.n_head, self.head_dim
        Q = self.q_proj(q).view(B, Nq, H, D).transpose(1, 2)   # (B, H, Nq, D)
        K = self.k_proj(kv).view(B, Nk, H, D).transpose(1, 2)  # (B, H, Nk, D)
        V = self.v_proj(kv).view(B, Nk, H, D).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) * (D ** -0.5)          # (B, H, Nq, Nk)
        if attn_mask is not None:
            attn = attn + attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, H * D)
        return self.out_proj(out)


class Block(nn.Module):
    """Time-conditioned transformer block"""

    def __init__(
        self,
        n_embd=512,
        n_head=8,
        dim_feedforward=2048,
        dropout=0.1,
        activate="GELU",
        num_timesteps=1000,
        timestep_type="adalayernorm_abs",
        attn_type="self",
        num_labels=None,  # attn_type = 'selfcondition'
        label_type="adalayernorm",  # attn_type = 'selfcondition'
        cond_emb_dim=None,  # attn_type = 'selfcross'
        mlp_type="fc",
    ):
        super().__init__()
        self.attn_type = attn_type

        if "adalayernorm" in timestep_type:
            self.ln1 = AdaLayerNorm(n_embd, num_timesteps, timestep_type)
        elif "adainnorm" in timestep_type:
            self.ln1 = AdaInsNorm(n_embd, num_timesteps, timestep_type)
        else:
            raise ValueError(f"timestep_type={timestep_type} not valid.")

        if attn_type == "self":
            self.attn = SelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
            self.ln2 = nn.LayerNorm(n_embd)
        elif attn_type == "selfcondition":  # conditioned on int labels
            self.attn = SelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
            if "adalayernorm" in label_type:
                self.ln2 = AdaLayerNorm(n_embd, num_labels, label_type)
            else:
                self.ln2 = AdaInsNorm(n_embd, num_labels, label_type)
        elif attn_type == "selfcross":  # cross attention with cond_emb
            self.attn1 = SelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout)
            self.attn2 = CrossAttention(
                n_embd=n_embd,
                n_head=n_head,
                dropout=dropout,
                kv_embd=cond_emb_dim,
            )
            if "adalayernorm" in timestep_type:
                self.ln1_1 = AdaLayerNorm(n_embd, num_timesteps, timestep_type)
            else:
                raise ValueError(f"timestep_type={timestep_type} not valid.")
            self.ln2 = nn.LayerNorm(n_embd)
        else:
            raise ValueError(f"attn_type={attn_type} not valid.")

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()
        if mlp_type == "fc":
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, dim_feedforward),
                act,
                nn.Linear(dim_feedforward, n_embd),
                nn.Dropout(dropout),
            )
        else:
            raise NotImplemented

    def forward(self, x, timestep, cond_output=None, mask=None):
        if self.attn_type == "self":
            x = x + self.attn(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.mlp(self.ln2(x))
        elif self.attn_type == "selfcondition":
            x = x + self.attn(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.mlp(self.ln2(x, cond_output))
        elif self.attn_type == "selfcross":
            x = x + self.attn1(self.ln1(x, timestep), attn_mask=mask)
            x = x + self.attn2(self.ln1_1(x, timestep), cond_output, attn_mask=mask)
            x = x + self.mlp(self.ln2(x))
        else:
            return NotImplemented
        return x


class DenoiseTransformer(nn.Module):
    """Base denoising transformer class"""

    def __init__(
        self,
        n_layer=4,
        n_embd=512,
        n_head=8,
        dim_feedforward=2048,
        dropout=0.1,
        activate="GELU",
        num_timesteps=1000,
        timestep_type="adalayernorm_abs",
        context_dim=256,
        mlp_type="fc",
    ):
        super().__init__()

        # transformer backbone
        if context_dim == 0:
            self.tf_blocks = nn.Sequential(
                *[
                    Block(
                        n_embd,
                        n_head,
                        dim_feedforward,
                        dropout,
                        activate,
                        num_timesteps,
                        timestep_type,
                        mlp_type=mlp_type,
                        attn_type="self",
                    )
                    for _ in range(n_layer)
                ]
            )
        else:
            self.tf_blocks = nn.Sequential(
                *[
                    Block(
                        n_embd,
                        n_head,
                        dim_feedforward,
                        dropout,
                        activate,
                        num_timesteps,
                        timestep_type,
                        mlp_type=mlp_type,
                        attn_type="selfcross",
                        cond_emb_dim=context_dim,
                    )
                    for _ in range(n_layer)
                ]
            )

    @staticmethod
    def _encoder_mlp(hidden_size, input_size):
        mlp_layers = [
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        ]
        return nn.Sequential(*mlp_layers)

    @staticmethod
    def _decoder_mlp(hidden_size, output_size):
        mlp_layers = [
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        ]
        return nn.Sequential(*mlp_layers)


class MIDiffusionContinuous(DenoiseTransformer):
    """Continuous denoising transformer network where all object properties are
    treated as continuous"""

    def __init__(
        self,
        network_dim,
        seperate_all=True,
        n_layer=4,
        n_embd=512,
        n_head=8,
        dim_feedforward=2048,
        dropout=0.1,
        activate="GELU",
        num_timesteps=1000,
        timestep_type="adalayernorm_abs",
        context_dim=256,
        mlp_type="fc",
    ):
        # initialize self.tf_blocks, the transformer backbone
        super().__init__(
            n_layer,
            n_embd,
            n_head,
            dim_feedforward,
            dropout,
            activate,
            num_timesteps,
            timestep_type,
            context_dim,
            mlp_type,
        )

        # feature dimensions
        self.objectness_dim, self.class_dim, self.objfeat_dim = (
            network_dim["objectness_dim"],
            network_dim["class_dim"],
            network_dim["objfeat_dim"],
        )
        self.translation_dim, self.size_dim, self.angle_dim = (
            network_dim["translation_dim"],
            network_dim["size_dim"],
            network_dim["angle_dim"],
        )
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.channels = (
            self.bbox_dim + self.objectness_dim + self.class_dim + self.objfeat_dim
        )

        # Initial feature specific processing
        self.seperate_all = seperate_all
        if self.seperate_all:
            self.bbox_embedf = self._encoder_mlp(n_embd, self.bbox_dim)
            self.bbox_hidden2output = self._decoder_mlp(n_embd, self.bbox_dim)
            feature_str = "translation/size/angle"

            if self.class_dim > 0:
                self.class_embedf = self._encoder_mlp(n_embd, self.class_dim)
                feature_str += "/class"
            if self.objectness_dim > 0:
                self.objectness_embedf = self._encoder_mlp(n_embd, self.objectness_dim)
                feature_str += "/objectness"
            if self.objfeat_dim > 0:
                self.objfeat_embedf = self._encoder_mlp(n_embd, self.objfeat_dim)
                feature_str += "/objfeat"
            # print("separate unet1d encoder/decoder of {}".format(feature_str))
        else:
            self.init_mlp = self._encoder_mlp(n_embd, self.channels)
            print("unet1d encoder of all object properties")

        # Final feature specific processing
        if self.seperate_all:
            self.bbox_hidden2output = self._decoder_mlp(n_embd, self.bbox_dim)
            if self.class_dim > 0:
                self.class_hidden2output = self._decoder_mlp(n_embd, self.class_dim)
            if self.objectness_dim > 0:
                self.objectness_hidden2output = self._decoder_mlp(
                    n_embd, self.objectness_dim
                )
            if self.objfeat_dim > 0:
                self.objfeat_hidden2output = self._decoder_mlp(n_embd, self.objfeat_dim)
        else:
            self.hidden2output = self._decoder_mlp(n_embd, self.channels)

    def forward(self, x, time, context=None, context_cross=None):
        # x: (B, N, C)
        if context_cross is not None:
            raise NotImplemented  # TODO

        # initial processing
        if self.seperate_all:
            x_bbox = self.bbox_embedf(x[:, :, 0 : self.bbox_dim])

            if self.class_dim > 0:
                start_index = self.bbox_dim
                x_class = self.class_embedf(
                    x[:, :, start_index : start_index + self.class_dim]
                )
            else:
                x_class = 0

            if self.objectness_dim > 0:
                start_index = self.bbox_dim + self.class_dim
                x_object = self.objectness_embedf(
                    x[:, :, start_index : start_index + self.objectness_dim]
                )
            else:
                x_object = 0

            if self.objfeat_dim > 0:
                start_index = self.bbox_dim + self.class_dim + self.objectness_dim
                x_objfeat = self.objfeat_embedf(
                    x[:, :, start_index : start_index + self.objfeat_dim]
                )
            else:
                x_objfeat = 0

            x = x_bbox + x_class + x_object + x_objfeat
        else:
            x = self.init_mlp(x)

        # transformer
        for block_idx in range(len(self.tf_blocks)):
            x = self.tf_blocks[block_idx](x, time, context)

        # final processing
        if self.seperate_all:
            out = self.bbox_hidden2output(
                x
            )  # b, 12, 12 but should be # b, 12, 512 -> b, 12, 8 TODO:
            assert out.shape[2] == 8
            if self.class_dim > 0:
                out_class = self.class_hidden2output(
                    x
                )  # b, 12, 18 but should be b, 12, 22
                assert out_class.shape[2] == 22
                out = torch.cat([out, out_class], dim=2).contiguous()
            if self.objectness_dim > 0:
                out_object = self.objectness_hidden2output(x)
                out = torch.cat([out, out_object], dim=2).contiguous()
            if self.objfeat_dim > 0:
                out_objfeat = self.objfeat_hidden2output(x)
                out = torch.cat([out, out_objfeat], dim=2).contiguous()
        else:
            out = self.hidden2output(x)

        return out  # b, 12, 30


class SceneDiffuserMiDiffusion(nn.Module):
    """
    Scene diffusion on a set of un-ordered objects. The number of objects and types
    of objects are not fixed. The object vectors consist of [translation, rotation,
    model_vector]. All scenes have `max_num_objects_per_scene` objects.

    This implements the continuous baseline model from MiDiffusion:
    https://arxiv.org/abs/2405.21066
    """

    def __init__(self):
        super().__init__()
        self.floor_encoder, floor_cond_dim = load_floor_encoder_from_config()
        context_dim = floor_cond_dim  # 64D from PointNet
        # print(f"[Ashok] Using floor encoder with context dim: {context_dim}")


        network_dim = {
            "objectness_dim": 0,  # Not used by our scene representation
            "class_dim": 22,
            "translation_dim": 3,
            "size_dim": 3,
            "angle_dim": 2,
            "objfeat_dim": 0,  # Not used by our scene representation
        }
        self.model = MIDiffusionContinuous(
            network_dim=network_dim,
            seperate_all=True,
            n_layer=8,
            n_embd=512,
            n_head=4,
            dim_feedforward=2048,
            dropout=0.1,
            activate='GELU',
            timestep_type='adalayernorm_abs',
            context_dim=context_dim,
            mlp_type='fc',
        )

    def predict_noise(
        self,
        noisy_scenes: torch.Tensor,
        timesteps: Union[torch.IntTensor, int],
        fpbpn,
    ) -> torch.Tensor:
        model = self.model

        floor_cond = self.floor_encoder(
            fpbpn.to(noisy_scenes.dtype)
        )  # Shape (B, 64)
        floor_cond = floor_cond.to(noisy_scenes.dtype)

        context = floor_cond.unsqueeze(1).expand(
            -1, noisy_scenes.size(1), -1
        )  # Shape (B, N, 64)
        predicted_noise = model(
            noisy_scenes, time=timesteps, context=context, context_cross=None
        )  # Shape (B, N, V)
        return predicted_noise


if __name__ == "__main__":
    model = SceneDiffuserMiDiffusion()
    scenes_shape = (4, 12, 30)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    noise = torch.randn(scenes_shape).to(device)
    timesteps = torch.randint(0, 1000, (scenes_shape[0],)).to(device)
    fpbpn = torch.randn((scenes_shape[0], 256, 4)).to(device)
    predicted_noise = model.predict_noise(noise, timesteps, fpbpn)
    print(predicted_noise.shape)


