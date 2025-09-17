import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Try to import the custom CUDA trilinear kernel; fall back gracefully if unavailable.
try:
    import trilinear as _trilinear  # pybind extension shipped with the original repo
    _HAS_TRILINEAR = True
except Exception:
    _HAS_TRILINEAR = False

from utils.LUT import *
import net


# -----------------------------
# Utilities
# -----------------------------
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C = size[:2]
    feat_var = feat.reshape(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.reshape(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content, style):
        assert content.size()[:2] == style.size()[:2]
        size = content.size()
        style_mean, style_std = calc_mean_std(style)
        content_mean, content_std = calc_mean_std(content)
        normalized_feat = (content - content_mean.expand(size)) / (content_std.expand(size))
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2  # "same" spatial size after padding
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        nn.init.normal_(self.conv2d.weight, mean=0, std=0.5)
        self.bn = nn.BatchNorm2d(out_channels)
        nn.init.normal_(self.bn.weight, mean=0, std=0.5)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.bn(out)
        return out


class SplattingBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SplattingBlock2, self).__init__()
        self.conv1 = ConvLayer(in_channels, in_channels, 3, 1)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1)
        self.adain = AdaIN()

    def forward(self, c, s):
        c1 = torch.tanh(self.conv1(c))
        c = torch.tanh(self.conv2(c1 + c))
        s1 = torch.tanh(self.conv1(s))
        s = torch.tanh(self.conv2(s1 + s))
        sed = self.adain(c, s)
        return sed


# -----------------------------
# MPS-safe AdaptiveAvgPool2d
# -----------------------------
class SafeAdaptiveAvgPool2d(nn.Module):
    """
    MPS-safe substitute for AdaptiveAvgPool2d.
    On MPS, when input sizes are not divisible by target output, falls back to
    bilinear resize with align_corners=False (not 'area', which re-enters adaptive pooling).
    """
    def __init__(self, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.out_h, self.out_w = output_size
        self.native = nn.AdaptiveAvgPool2d((self.out_h, self.out_w))

    def forward(self, x: torch.Tensor):
        if x.device.type == "mps":
            H, W = x.shape[-2], x.shape[-1]
            if (H % self.out_h != 0) or (W % self.out_w != 0):
                # Avoid 'area' (it calls adaptive_avg_pool2d under the hood on MPS).
                return F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear")
        return self.native(x)



# -----------------------------
# NLUT Model
# -----------------------------
class NLUTNet(nn.Module):
    def __init__(self, nsw, dim, *args, **kwargs):
        super(NLUTNet, self).__init__()
        vgg = net.vgg
        vgg.load_state_dict(torch.load('models/vgg_normalised.pth', weights_only=False))
        self.encoder = net.Net(vgg)
        self.encoder.eval()
        self.adain = AdaIN()

        self.SB2 = SplattingBlock2(64, 256)
        self.SB3 = SplattingBlock2(128, 256)
        self.SB4 = SplattingBlock2(256, 256)
        self.SB5 = SplattingBlock2(512, 256)

        # MPS-safe pooling to fixed 3x3 per stage
        self.pg5 = SafeAdaptiveAvgPool2d(3)
        self.pg4 = SafeAdaptiveAvgPool2d(3)
        self.pg3 = SafeAdaptiveAvgPool2d(3)
        self.pg2 = SafeAdaptiveAvgPool2d(3)

        self.pre = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        nsw = nsw.split("+")
        num, s, w = int(nsw[0]), int(nsw[1]), int(nsw[2])
        self.CLUTs = CLUT(num, dim, s, w)
        self.TrilinearInterpolation = TrilinearInterpolation()
        last_channel = 256 * 4  # concat of four 256-channel pooled maps

        # Keep the first classifier conv as 3x3 (matches common checkpoints)
        # 3x3, stride=2 on 3x3 input -> 1x1 output spatially.
        self.classifier = nn.Sequential(
            nn.Conv2d(last_channel, 512, 3, 2),
            nn.BatchNorm2d(512),

            nn.Tanh(),
            nn.Conv2d(512, 512 * 2, 1, 1),
            nn.BatchNorm2d(512 * 2),

            nn.Tanh(),
            nn.Conv2d(512 * 2, 512, 1, 1),
            nn.BatchNorm2d(512),

            nn.Tanh(),
            nn.Conv2d(512, num, 1, 1),
            nn.BatchNorm2d(num),
        )

    def forward(self, img, img_org, style, TVMN=None):
        content = img
        B, C, H, W = content.size()
        content = self.pre(content)
        style = self.pre(style)

        resize_style = F.interpolate(style, (256, 256), mode='bilinear', align_corners=False)
        resize_content = F.interpolate(content, (256, 256), mode='bilinear', align_corners=False)

        style_feats = self.encoder.encode_with_intermediate(resize_style)
        content_feat = self.encoder.encode_with_intermediate(resize_content)

        stylized5 = self.SB5(content_feat[-1], style_feats[-1])  # [B,256,16,16] -> pooled to 3x3
        stylized4 = self.SB4(content_feat[-2], style_feats[-2])
        stylized3 = self.SB3(content_feat[-3], style_feats[-3])
        stylized2 = self.SB2(content_feat[-4], style_feats[-4])

        stylized5 = self.pg5(stylized5)  # -> [B,256,3,3]
        stylized4 = self.pg4(stylized4)  # -> [B,256,3,3]
        stylized3 = self.pg3(stylized3)  # -> [B,256,3,3]
        stylized2 = self.pg2(stylized2)  # -> [B,256,3,3]

        stylized1 = torch.cat((stylized2, stylized3, stylized4, stylized5), dim=1)  # [B, 1024, 3,3]
        pred = self.classifier(stylized1)[:, :, 0, 0]  # [B, num]

        D3LUT, tvmn = self.CLUTs(pred, TVMN)
        img_out = self.TrilinearInterpolation(D3LUT, img_org)
        img_out = img_out + img_org  # residual

        output = img_out
        return img_out, output, {
            "LUT": D3LUT,
            "tvmn": tvmn,
        }


# -----------------------------
# CLUT and helpers
# -----------------------------
class CLUT(nn.Module):
    def __init__(self, num, dim=33, s="-1", w="-1", *args, **kwargs):
        super(CLUT, self).__init__()
        self.num = num
        self.dim = dim
        self.s, self.w = s, w = eval(str(s)), eval(str(w))
        # +: compressed;  -: uncompressed
        if s == -1 and w == -1:  # standard 3DLUT
            self.mode = '--'
            self.LUTs = nn.Parameter(torch.zeros(num, 3, dim, dim, dim))
        elif s != -1 and w == -1:
            self.mode = '+-'
            self.s_Layers = nn.Parameter(torch.rand(dim, s) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(s, num * 3 * dim * dim))
        elif s == -1 and w != -1:
            self.mode = '-+'
            self.w_Layers = nn.Parameter(torch.rand(w, dim * dim) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(num * 3 * dim, w))
        else:  # full-version CLUT
            self.mode = '++'
            self.s_Layers = nn.Parameter(torch.rand(dim, s) / 5 - 0.1)
            self.w_Layers = nn.Parameter(torch.rand(w, dim * dim) / 5 - 0.1)
            self.LUTs = nn.Parameter(torch.zeros(s * num * 3, w))
        print("n=%d s=%d w=%d" % (num, s, w), self.mode)

    def reconstruct_luts(self):
        dim = self.dim
        num = self.num
        if self.mode == "--":
            D3LUTs = self.LUTs
        else:
            if self.mode == "+-":
                # d,s  x  s,num*3dd  -> d,num*3dd -> d,num*3,dd -> num,3,d,dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs).reshape(dim, num * 3, dim * dim) \
                    .permute(1, 0, 2).reshape(num, 3, self.dim, self.dim, self.dim)
            if self.mode == "-+":
                # num*3d,w x w,dd -> num*3d,dd -> num,3ddd
                CUBEs = self.LUTs.mm(self.w_Layers).reshape(num, 3, self.dim, self.dim, self.dim)
            if self.mode == "++":
                # s*num*3, w  x   w, dd -> s*num*3,dd -> s,num*3*dd -> d,num*3*dd -> num,-1
                CUBEs = self.s_Layers.mm(self.LUTs.mm(self.w_Layers).reshape(-1, num * 3 * dim * dim)) \
                    .reshape(dim, num * 3, dim ** 2).permute(1, 0, 2).reshape(num, 3, self.dim, self.dim, self.dim)
            D3LUTs = cube_to_lut(CUBEs)

        return D3LUTs

    def combine(self, weight, TVMN):  # n,num
        dim = self.dim
        num = self.num

        D3LUTs = self.reconstruct_luts()
        if TVMN is None:
            tvmn = 0
        else:
            tvmn = TVMN(D3LUTs)
        D3LUT = weight.mm(D3LUTs.reshape(num, -1)).reshape(-1, 3, dim, dim, dim)
        return D3LUT, tvmn

    def forward(self, weight, TVMN=None):
        lut, tvmn = self.combine(weight, TVMN)
        return lut, tvmn


class BackBone(nn.Module):
    def __init__(self, last_channel=128, ):
        super(BackBone, self).__init__()
        ls = [
            *discriminator_block(3, 16, normalization=True),   # 128**16
            *discriminator_block(16, 32, normalization=True),  # 64**32
            *discriminator_block(32, 64, normalization=True),  # 32**64
            *discriminator_block(64, 128, normalization=True), # 16**128
            *discriminator_block(128, last_channel, normalization=False), # 8**128
            nn.Dropout(p=0.5),
            nn.AdaptiveAvgPool2d(1),
        ]
        self.model = nn.Sequential(*ls)

    def forward(self, x):
        return self.model(x)


class TVMN(nn.Module):  # (n,)3,d,d,d   or   (n,)3,d
    def __init__(self, dim=33):
        super(TVMN, self).__init__()
        self.dim = dim
        self.relu = torch.nn.ReLU()

        weight_r = torch.ones(1, 1, dim, dim, dim - 1, dtype=torch.float)
        weight_r[..., (0, dim - 2)] *= 2.0
        weight_g = torch.ones(1, 1, dim, dim - 1, dim, dtype=torch.float)
        weight_g[..., (0, dim - 2), :] *= 2.0
        weight_b = torch.ones(1, 1, dim - 1, dim, dim, dtype=torch.float)
        weight_b[..., (0, dim - 2), :, :] *= 2.0
        self.register_buffer('weight_r', weight_r, persistent=False)
        self.register_buffer('weight_g', weight_g, persistent=False)
        self.register_buffer('weight_b', weight_b, persistent=False)

        self.register_buffer('tvmn_shape', torch.empty(3), persistent=False)

    def forward(self, LUT):
        dim = self.dim
        tvmn = 0 + self.tvmn_shape
        if len(LUT.shape) > 3:  # n,3,d,d,d  or  3,d,d,d
            dif_r = LUT[..., :-1] - LUT[..., 1:]
            dif_g = LUT[..., :-1, :] - LUT[..., 1:, :]
            dif_b = LUT[..., :-1, :, :] - LUT[..., 1:, :, :]
            tvmn[0] = torch.mean(dif_r ** 2 * self.weight_r[:, 0]) + \
                      torch.mean(dif_g ** 2 * self.weight_g[:, 0]) + \
                      torch.mean(dif_b ** 2 * self.weight_b[:, 0])
            tvmn[1] = torch.mean(self.relu(dif_r * self.weight_r[:, 0]) ** 2) + \
                      torch.mean(self.relu(dif_g * self.weight_g[:, 0]) ** 2) + \
                      torch.mean(self.relu(dif_b * self.weight_b[:, 0]) ** 2)
            tvmn[2] = 0
        else:  # n,3,d  or  3,d
            dif = LUT[..., :-1] - LUT[..., 1:]
            tvmn[1] = torch.mean(self.relu(dif))
            dif = dif ** 2
            dif[..., (0, dim - 2)] *= 2.0
            tvmn[0] = torch.mean(dif)
            tvmn[2] = 0
        return tvmn


def discriminator_block(in_filters, out_filters, kernel_size=3, sp="2_1", normalization=False):
    stride = int(sp.split("_")[0])
    padding = int(sp.split("_")[1])

    layers = [
        nn.Conv2d(in_filters, out_filters, kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))

    return layers


# -----------------------------
# Trilinear (CUDA + pure PyTorch fallback)
# -----------------------------
class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):
        if not _HAS_TRILINEAR:
            raise RuntimeError("Custom 'trilinear' CUDA extension not available.")

        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        if batch == 1:
            assert 1 == _trilinear.forward(lut,
                                           x,
                                           output,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           batch)
        elif batch > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == _trilinear.forward(lut,
                                           x.permute(1, 0, 2, 3).contiguous(),
                                           output,
                                           dim,
                                           shift,
                                           binsize,
                                           W,
                                           H,
                                           batch)
            output = output.permute(1, 0, 2, 3).contiguous()

        # Save small metadata tensors on the SAME device as x
        int_package = torch.tensor([dim, shift, W, H, batch], dtype=torch.int32, device=x.device)
        float_package = torch.tensor([binsize], dtype=torch.float32, device=x.device)
        ctx.save_for_backward(lut, x, int_package, float_package)
        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        if not _HAS_TRILINEAR:
            raise RuntimeError("Custom 'trilinear' CUDA extension not available.")

        lut, x, int_package, float_package = ctx.saved_tensors
        dim, shift, W, H, batch = [int(v.item()) for v in int_package]
        binsize = float(float_package[0].item())

        if batch == 1:
            assert 1 == _trilinear.backward(x,
                                            x_grad,
                                            lut_grad,
                                            dim,
                                            shift,
                                            binsize,
                                            W,
                                            H,
                                            batch)
        elif batch > 1:
            assert 1 == _trilinear.backward(x.permute(1, 0, 2, 3).contiguous(),
                                            x_grad.permute(1, 0, 2, 3).contiguous(),
                                            lut_grad,
                                            dim,
                                            shift,
                                            binsize,
                                            W,
                                            H,
                                            batch)
        return lut_grad, x_grad


def _apply_lut_trilinear_torch(lut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch trilinear LUT application.
    lut: [1 or B or K, 3, D, D, D] ; x: [B, 3, H, W] in [0,1]
    Works on CPU/MPS/CUDA (no custom kernels). Differentiable.
    """
    B, C, H, W = x.shape
    assert C == 3, "Expect 3-channel RGB"
    device = x.device
    dtype = x.dtype

    if lut.dim() == 5 and lut.size(0) == 1:
        lut = lut.expand(B, -1, -1, -1, -1)
    elif lut.dim() == 5 and lut.size(0) != B:
        # If K LUTs provided (K != B), just broadcast first to all images
        lut = lut[0:1].expand(B, -1, -1, -1, -1)

    D = lut.shape[-1]
    # Normalize to [0, D-1]
    v = (x.clamp(0, 1) * (D - 1)).permute(0, 2, 3, 1)  # [B,H,W,3], dtype x
    i0 = torch.floor(v).to(torch.long)
    i1 = (i0 + 1).clamp(max=D - 1)
    f = v - i0.to(v.dtype)  # [B,H,W,3]

    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg, fb = f[..., 0], f[..., 1], f[..., 2]

    b_idx = torch.arange(B, device=device)[:, None, None].expand(B, H, W)

    def gather(rr, gg, bb):
        """
        Advanced indexing with multiple index tensors returns [B,H,W,3].
        We permute to channels-first [B,3,H,W] for broadcasting with weights.
        """
        out = lut[b_idx, :, rr, gg, bb]  # shape: [B, H, W, 3]
        return out.permute(0, 3, 1, 2).contiguous()  # -> [B, 3, H, W]

    c000 = gather(r0, g0, b0)
    c100 = gather(r1, g0, b0)
    c010 = gather(r0, g1, b0)
    c110 = gather(r1, g1, b0)
    c001 = gather(r0, g0, b1)
    c101 = gather(r1, g0, b1)
    c011 = gather(r0, g1, b1)
    c111 = gather(r1, g1, b1)

    w000 = (1 - fr) * (1 - fg) * (1 - fb)
    w100 = (fr) * (1 - fg) * (1 - fb)
    w010 = (1 - fr) * (fg) * (1 - fb)
    w110 = (fr) * (fg) * (1 - fb)
    w001 = (1 - fr) * (1 - fg) * (fb)
    w101 = (fr) * (1 - fg) * (fb)
    w011 = (1 - fr) * (fg) * (fb)
    w111 = (fr) * (fg) * (fb)

    out = (
        c000 * w000.unsqueeze(1)
        + c100 * w100.unsqueeze(1)
        + c010 * w010.unsqueeze(1)
        + c110 * w110.unsqueeze(1)
        + c001 * w001.unsqueeze(1)
        + c101 * w101.unsqueeze(1)
        + c011 * w011.unsqueeze(1)
        + c111 * w111.unsqueeze(1)
    )  # [B,3,H,W], dtype matches x
    return out


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self, mo=False, clip=False):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        # Non-CUDA (CPU/MPS): use pure PyTorch path
        if x.device.type != "cuda" or not _HAS_TRILINEAR:
            if lut.dim() == 5 and lut.size(0) > 1 and lut.size(0) != x.size(0):
                # Return [B, K, 3, H, W] when multiple LUTs given (mirror original semantics)
                B, C, H, W = x.shape
                K = lut.size(0)
                res = torch.empty(B, K, C, H, W, device=x.device, dtype=x.dtype)
                for i in range(K):
                    res[:, i] = _apply_lut_trilinear_torch(lut[i:i + 1], x)
                return res
            return _apply_lut_trilinear_torch(lut, x)

        # CUDA: use the custom op (original behavior), but allocate on x.device
        if lut.shape[0] > 1:
            if lut.shape[0] == x.shape[0]:  # per-image LUT
                res = torch.empty_like(x)
                for i in range(lut.shape[0]):
                    res[i:i + 1] = TrilinearInterpolationFunction.apply(lut[i:i + 1], x[i:i + 1])[1]
            else:
                n, c, h, w = x.shape
                res = torch.empty(n, lut.shape[0], c, h, w, device=x.device, dtype=x.dtype)
                for i in range(lut.shape[0]):
                    res[:, i] = TrilinearInterpolationFunction.apply(lut[i:i + 1], x)[1]
        else:  # one LUT for the whole batch
            res = TrilinearInterpolationFunction.apply(lut, x)[1]
        return res
