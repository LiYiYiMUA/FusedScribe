# Wu et al., TIP 2017 spatial JND in PyTorch (training-time heatmap)
# Interface-compatible with your previous JND class / apply_jnd_embed().

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi

class JND(nn.Module):
    """
    TIP 2017: Enhanced JND with Pattern Complexity (Wu et al.)
    - Inputs expected in [0,1] after preprocess; internally scaled to [0,255].
    - heatmaps() returns a mask in [0,1] with shape (B, out_channels, H, W).
    - Channel mapping: in_channels=1 => Y from RGB; out_channels=3 => repeat, optional blue bias.
    """

    def __init__(
        self,
        preprocess = lambda x: x,
        postprocess = lambda x: x,
        in_channels: int = 1,
        out_channels: int = 3,
        blue: bool = False,
        # pattern complexity settings
        orient_step_deg: float = 12.0,   # T in degrees (default per paper)
        pattern_size: int = 3,           # local receptive field (3x3 as used in paper's experiments)
        # NAMM overlap coefficient (C in paper)
        namm_c: float = 0.3,
        # contrast masking params (alpha, beta) in Eq.(11)
        cm_alpha: float = 16.0,
        cm_beta: float = 26.0,
        # pattern masking params (a1, a2, a3) in Eq.(10)
        pm_a1: float = 0.8,
        pm_a2: float = 2.7,
        pm_a3: float = 0.1,
        # kernel for background luminance (LA); keep compatible with your previous kernel
    ):
        super().__init__()
        assert in_channels in (1, 3) and out_channels in (1, 3)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.blue = blue

        self.orient_step_deg = orient_step_deg
        self.pattern_size = pattern_size
        self.namm_c = namm_c
        self.cm_alpha = cm_alpha
        self.cm_beta = cm_beta
        self.pm_a1 = pm_a1
        self.pm_a2 = pm_a2
        self.pm_a3 = pm_a3

        # ---------- Prewitt edge kernels (paper uses Prewitt) ----------
        kx = torch.tensor([[-1., 0., 1.],
                           [-1., 0., 1.],
                           [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor([[ 1.,  1.,  1.],
                           [ 0.,  0.,  0.],
                           [-1., -1., -1.]]).unsqueeze(0).unsqueeze(0)

        # Background luminance kernel (compatible with your previous code; sum = 32)
        kl = torch.tensor([[1., 1., 1., 1., 1.],
                           [1., 2., 2., 2., 1.],
                           [1., 2., 0., 2., 1.],
                           [1., 2., 2., 2., 1.],
                           [1., 1., 1., 1., 1.]]).unsqueeze(0).unsqueeze(0)

        self.conv_x = nn.Conv2d(1, 1, 3, 1, 1, bias=False)  # horizontal gradient
        self.conv_y = nn.Conv2d(1, 1, 3, 1, 1, bias=False)  # vertical gradient
        self.conv_lum = nn.Conv2d(1, 1, 5, 1, 2, bias=False)

        with torch.no_grad():
            self.conv_x.weight.copy_(kx)
            self.conv_y.weight.copy_(ky)
            self.conv_lum.weight.copy_(kl)

        for p in self.parameters():
            p.requires_grad_(False)

        self.preprocess = preprocess
        self.postprocess = postprocess

        # constants
        self.register_buffer("_rgb2y", torch.tensor([0.299, 0.587, 0.114]).view(1,3,1,1), persistent=False)
        self.eps = 1e-6

    # ---------------------- Components ----------------------
    def _to_luma_255(self, imgs01: torch.Tensor) -> torch.Tensor:
        """Map RGB[0,1] -> Y[0,255] if in_channels==1, else pass-through to [0,255]."""
        x = imgs01 * 255.0
        if self.in_channels == 1:
            # luminance from RGB
            y = (x * self._rgb2y.to(x.device)).sum(dim=1, keepdim=True)
            return y
        else:
            # already 1-channel (treat as Y) or 3-channel (process each? paper is per Y; keep as 3→per-channel)
            return x

    def jnd_la(self, y255: torch.Tensor) -> torch.Tensor:
        """Luminance adaptation LA, Eq.(13). Expect y in [0,255], shape (B,1,H,W) or (B,C,H,W)."""
        B, C, H, W = y255.shape
        # background luminance B(x): mean-like filter (compatible with your prior implementation)
        Bkg = self.conv_lum(y255) / 32.0
        la = torch.empty_like(Bkg)
        mask = (Bkg < 127.0)
        # below 127
        la[mask] = 17.0 * (1.0 - torch.sqrt(torch.clamp(Bkg[mask] / 127.0, min=0.0) + self.eps))
        # >= 127
        la[~mask] = 3.0 * (Bkg[~mask] - 127.0) / 128.0 + 3.0
        return la  # units ~ intensity

    def _prewitt_grad(self, y255_1ch: torch.Tensor):
        gx = self.conv_x(y255_1ch)
        gy = self.conv_y(y255_1ch)
        return gx, gy

    def _luminance_contrast(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        """C_l = sqrt(Gx^2 + Gy^2), Eq.(7)."""
        return torch.sqrt(gx*gx + gy*gy + self.eps)

    def _orientation_deg(self, gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
        """θ in [0,180). atan2(gy, gx) ∈ (-180,180], then mod 180."""
        theta = torch.atan2(gy, gx) * (180.0 / pi)
        theta = torch.remainder(theta, 180.0)  # wrap to [0,180)
        return theta

    def _pattern_complexity(self, theta_deg: torch.Tensor) -> torch.Tensor:
        """
        Cp(x): number of distinct orientation bins in a local KxK (pattern_size) window, Eq.(4)(6).
        Implementation: one-hot -> 2D max-pool -> sum over bins.
        """
        B, C, H, W = theta_deg.shape  # expect C==1
        assert C == 1, "pattern complexity expects 1-channel luminance orientation"
        T = self.orient_step_deg
        n_bins = int(round(180.0 / T))  # 180/12=15
        # quantize (nearest bin)
        theta_q = torch.floor(theta_deg / T + 0.5).long() % n_bins  # (B,1,H,W)
        # one-hot -> (B, H, W, n_bins) -> (B, n_bins, H, W)
        oh = F.one_hot(theta_q.squeeze(1), num_classes=n_bins).permute(0,3,1,2).to(theta_deg.dtype)
        # presence in local window via max_pool
        k = self.pattern_size
        pres = F.max_pool2d(oh, kernel_size=k, stride=1, padding=k//2)
        pres = (pres > 0.5).to(theta_deg.dtype)  # (B, n_bins, H, W)
        cp = pres.sum(dim=1, keepdim=True)  # (B,1,H,W), range [1..n_bins]
        return cp

    def _pattern_masking(self, Cl: torch.Tensor, Cp: torch.Tensor) -> torch.Tensor:
        """MP = log2(1+Cl) * a1 * Cp^a2 / (Cp^2 + a3^2), Eq.(8)(9)(10)."""
        fCl = torch.log2(1.0 + Cl + self.eps)
        fCp = self.pm_a1 * (Cp ** self.pm_a2) / (Cp**2 + self.pm_a3**2)
        return fCl * fCp

    def _contrast_masking(self, Cl: torch.Tensor) -> torch.Tensor:
        """MC = 0.115 * alpha * Cl^2.4 / (Cl^2 + beta^2), Eq.(11)."""
        num = (Cl ** 2.4)
        den = (Cl ** 2) + (self.cm_beta ** 2)
        return 0.115 * self.cm_alpha * num / (den + self.eps)

    # ---------------------- Public API ----------------------
    @torch.no_grad()
    def heatmaps(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs in [0,1], shape (B,3,H,W) typically.
        return JND heat-map in [0,1], shape (B,out_channels,H,W)
        """
        # to luminance in [0,255]
        y255 = self._to_luma_255(imgs)

        # LA
        la = self.jnd_la(y255)

        # gradient & luminance contrast
        if y255.shape[1] != 1:
            # if 3-ch, compute on luminance anyway
            y_for_grad = (y255 * self._rgb2y.to(y255.device)).sum(dim=1, keepdim=True)
        else:
            y_for_grad = y255

        gx, gy = self._prewitt_grad(y_for_grad)
        Cl = self._luminance_contrast(gx, gy)

        # pattern complexity on luminance orientation
        theta = self._orientation_deg(gx, gy)
        Cp = self._pattern_complexity(theta)

        # MP & MC
        mp = self._pattern_masking(Cl, Cp)
        mc = self._contrast_masking(Cl)

        # spatial masking = max(MP, MC)
        ms = torch.maximum(mp, mc)

        # NAMM compose with LA
        jnd = la + ms - self.namm_c * torch.minimum(la, ms)
        jnd = torch.clamp_min(jnd, 0.0)  # keep non-negative

        # channel mapping
        if self.out_channels == 3:
            h = jnd.repeat(1, 3, 1, 1)
            if self.blue:
                h[:, 0] *= 0.5  # R
                h[:, 1] *= 0.5  # G
                # B unchanged
        else:
            # 1-channel
            if jnd.shape[1] != 1:
                h = jnd.mean(dim=1, keepdim=True)
            else:
                h = jnd

        # normalize to [0,1] (keep scale consistent with your apply_jnd_embed usage)
        return h / 255.0

    def forward(self, imgs: torch.Tensor, imgs_w: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Compatibility path (same as your old class):
        imgs, imgs_w in whatever range your preprocess expects (default pass-through).
        """
        x = self.preprocess(imgs)
        xw = self.preprocess(imgs_w)
        h = self.heatmaps(x)  # [0,1]
        out = x + alpha * h * (xw - x)
        return self.postprocess(out)


# === your embedding helper (unchanged API) ===
def apply_jnd_embed(base_img: torch.Tensor, proposed_img: torch.Tensor, alpha: float,
                    jnd_map_module: nn.Module, safe_to01=lambda x: x):
    """
    x_w = x_base + α · JND(x_base) · (x_prop - x_base)
    Inputs/outputs follow your pipeline. JND computed with no grad.
    """
    with torch.no_grad():
        hmap = jnd_map_module.heatmaps(safe_to01(base_img)).to(base_img.dtype)
    delta = proposed_img - base_img
    return base_img + alpha * hmap * delta
