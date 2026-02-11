import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.distributions.uniform import Uniform
import numpy as np
from timm.models.layers import DropPath, trunc_normal_

BNNorm3d = nn.BatchNorm3d
Activation = nn.GELU


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(ch_in, ch_out, 3, 1, 1, bias=False),
            BNNorm3d(ch_out),
            Activation(),
        )

    def forward(self, x):
        return self.up(x)


class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 3, 2, 1, bias=False),
            BNNorm3d(ch_out),
            Activation(),
        )

    def forward(self, x):
        return self.down(x)


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, groups=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, groups=groups, padding=1)
        self.bn1 = BNNorm3d(planes)
        self.act = Activation()

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, groups=groups, padding=1)
        self.bn2 = BNNorm3d(planes)

        self.need_proj = inplanes != planes
        if self.need_proj:
            self.down = nn.Sequential(
                nn.Conv3d(inplanes, planes, 1, 1),
                BNNorm3d(planes),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        if hasattr(self, "need_proj") and self.need_proj:
            identity = self.down(x)

        out = self.bn2(out) + identity
        return self.act(out)


class OPE(nn.Module):
    def __init__(self, inplanes, planes):
        super(OPE, self).__init__()
        self.conv = nn.Conv3d(inplanes, inplanes, 3, 1, padding=1)
        self.bn = BNNorm3d(inplanes)
        self.act = Activation()
        self.down = down_conv(inplanes, planes)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return self.down(x)


class local_block(nn.Module):
    def __init__(self, inplanes, hidden_planes, planes, down_or_up=None):
        super(local_block, self).__init__()
        if down_or_up is None:
            self.block = nn.Sequential(ResBlock(inplanes, hidden_planes))
        elif down_or_up == "down":
            self.block = nn.Sequential(
                ResBlock(inplanes, hidden_planes),
                down_conv(hidden_planes, planes),
            )
        elif down_or_up == "up":
            self.block = nn.Sequential(
                ResBlock(inplanes, hidden_planes),
                up_conv(hidden_planes, planes),
            )

    def forward(self, x):
        return self.block(x)


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool_size = pool_size
        self.pool = nn.AvgPool3d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        self.pool_ = nn.AvgPool3d(1, stride=1, padding=0, count_include_pad=False)
    def forward(self, x):
        # check all spatial dims (D,H,W)
        if min(x.shape[2:]) < self.pool_size:
            return self.pool_(x) - x
        return self.pool(x) - x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels):
        g = 8 
        super().__init__(g, num_channels)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = Activation()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class global_block(nn.Module):
    def __init__(self, in_dim, dim, num_heads, sr_ratio):
        super().__init__()
        self.proj = nn.Conv3d(in_dim, dim, 3, padding=1)
        self.norm1 = GroupNorm(dim)
        self.attn = Pooling(pool_size=3)
        self.norm2 = GroupNorm(dim)
        self.mlp = Mlp(dim, dim * 4, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class FuseConv1x1(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, 1, bias=False),
            GroupNorm(ch_out),
            Activation(),
        )

    def forward(self, x):
        return self.op(x)



class fUNet(nn.Module):
    def __init__(
        self,
        in_channel,
        num_classes,
        deep_supervised,
        layer_channel=[32, 64, 128, 256, 320],
        global_dim=[16, 32, 64, 128, 160],
        num_heads=[1, 2, 4, 8],
        sr_ratio=[8, 4, 2, 1],
    ):
        
        super(fUNet, self).__init__()

                
        self.deep_supervised = deep_supervised


        # ---------------- Encoder ----------------
        self.input_l0 = nn.Sequential(
            nn.Conv3d(in_channel, layer_channel[0], 3, 1, 1),
            BNNorm3d(layer_channel[0]),
            Activation(),
            nn.Conv3d(layer_channel[0], layer_channel[0], 3, 1, 1),
            BNNorm3d(layer_channel[0]),
            Activation(),
        )

        self.encoder1_l1_local = OPE(layer_channel[0], layer_channel[1])
        self.encoder1_l1_global = global_block(
            layer_channel[0], global_dim[0], num_heads[0], sr_ratio[0]
        )

        self.encoder1_l2_local = OPE(layer_channel[1], layer_channel[2])
        self.encoder1_l2_global = global_block(
            layer_channel[1], global_dim[1], num_heads[1], sr_ratio[1]
        )

        self.encoder1_l3_local = OPE(layer_channel[2], layer_channel[3])
        self.encoder1_l3_global = global_block(
            layer_channel[2], global_dim[2], num_heads[2], sr_ratio[2]
        )

        self.encoder1_l4_local = OPE(layer_channel[3], layer_channel[4])
        self.encoder1_l4_global = global_block(
            layer_channel[3], global_dim[3], num_heads[3], sr_ratio[3]
        )

        # ---------------- Decoder ----------------
        self.decoder1_l4_local = local_block(
            layer_channel[4], layer_channel[4], layer_channel[3], down_or_up="up"
        )
        self.decoder1_l4_global = global_block(
            layer_channel[4], global_dim[4], num_heads[3], sr_ratio[3]
        )

        self.fuse_d1_l3 = FuseConv1x1(layer_channel[3] + global_dim[3], layer_channel[3])
        self.decoder1_l3_local = local_block(
            layer_channel[3], layer_channel[3], layer_channel[2], down_or_up="up"
        )
        self.decoder1_l3_global = global_block(
            layer_channel[3], global_dim[2], num_heads[2], sr_ratio[2]
        )

        self.fuse_d1_l2 = FuseConv1x1(layer_channel[2] + global_dim[2], layer_channel[2])
        self.decoder1_l2_local = local_block(
            layer_channel[2], layer_channel[2], layer_channel[1], down_or_up="up"
        )
        self.decoder1_l2_global = global_block(
            layer_channel[2], global_dim[1], num_heads[1], sr_ratio[1]
        )

        self.fuse_d1_l1 = FuseConv1x1(layer_channel[1] + global_dim[1], layer_channel[1])
        self.decoder1_l1_local = local_block(
            layer_channel[1], layer_channel[1], layer_channel[0], down_or_up="up"
        )
        self.decoder1_l1_global = global_block(
            layer_channel[1], global_dim[0], num_heads[0], sr_ratio[0]
        )

        # ----------- Deep Supervision Outputs -------------

        self.out_l4 = nn.Conv3d(layer_channel[4], num_classes, 1)
        self.out_l3 = nn.Conv3d(layer_channel[3], num_classes, 1)
        self.out_l2 = nn.Conv3d(layer_channel[2], num_classes, 1)
        self.out_l1 = nn.Conv3d(layer_channel[1], num_classes, 1)

        # final output (highest resolution)
        self.output_final = nn.Conv3d(layer_channel[0], num_classes, 1)

        

    def _init_weights(self, m):
        # initialization
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = []

        # ---------------- Encoder ----------------
        x0 = self.input_l0(x)

        x1_local = self.encoder1_l1_local(x0)
        x0_global = self.encoder1_l1_global(x0)

        x2_local = self.encoder1_l2_local(x1_local)
        x1_global = self.encoder1_l2_global(x1_local)

        x3_local = self.encoder1_l3_local(x2_local)
        x2_global = self.encoder1_l3_global(x2_local)

        x4_local = self.encoder1_l4_local(x3_local)
        x3_global = self.encoder1_l4_global(x3_local)

     
        outputs.append(self.out_l4(x4_local))

        # ---------------- Decoder ----------------
        # level3
        x3_up_local = self.decoder1_l4_local(x4_local)
        x3_cat = torch.cat((x3_up_local, x3_global), 1)
        x3 = self.fuse_d1_l3(x3_cat)

        outputs.append(self.out_l3(x3))

        # level2
        x2_up_local = self.decoder1_l3_local(x3)
        x2_cat = torch.cat((x2_up_local, x2_global), 1)
        x2 = self.fuse_d1_l2(x2_cat)

        outputs.append(self.out_l2(x2))

        # level1
        x1_up_local = self.decoder1_l2_local(x2)
        x1_cat = torch.cat((x1_up_local, x1_global), 1)
        x1 = self.fuse_d1_l1(x1_cat)

        outputs.append(self.out_l1(x1))

        # final highest resolution
        x0_up = self.decoder1_l1_local(x1)
        out_final = self.output_final(x0_up)
        outputs.append(out_final)


        outputs = outputs[::-1]

        if self.deep_supervised:
            r = outputs
        else:
            r = outputs[0]
        return r





from torch._dynamo import OptimizedModule
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainer_fUNet(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True
        self.initial_lr = 1e-2
        self.num_epochs = 500

    def set_deep_supervision_enabled(self, enabled: bool):

        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        mod.deep_supervised = enabled

    @staticmethod
    def build_network_architecture(architecture_class_name,
                                   arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision):

        from dynamic_network_architectures.initialization.weight_init import InitWeights_He
        model = fUNet(in_channel=num_input_channels, num_classes=num_output_channels, deep_supervised=enable_deep_supervision)
        model.apply(InitWeights_He(1e-2))
        return model






