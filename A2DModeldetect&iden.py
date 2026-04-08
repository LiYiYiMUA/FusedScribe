# -*- coding: utf-8 -*-
# Copyright (c) Meta
# All rights reserved.

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Callable, Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as FbyV
import yaml
import lpips
import math
import matplotlib.pyplot as plt

# --- 项目内依赖 ---
import utils
import utils_img
import utils_model

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from omegaconf import OmegaConf

# videoseal 增强与指标
from videoseal_ori.videoseal.augmentation.augmenter import Augmenter
from torchmetrics.image.fid import FrechetInceptionDistance

# JND
from jndWAM import JND
from Jnd_fre import JND as JND_fre

# ----------------------- 全局 -----------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_eval = 0  # 仅用于命名

# 单例：JND
jnd_map = JND(in_channels=1, out_channels=3, blue=True).to(device).eval()
for p in jnd_map.parameters():
    p.requires_grad_(False)

# 频域 JND
jnd_map_fre = JND_fre(in_channels=1, out_channels=3, blue=False, freq=True, w_fm=0.05).to(device).eval()
for p in jnd_map_fre.parameters():
    p.requires_grad_(False)

# 单例：LPIPS
lpips_net = lpips.LPIPS(net='alex').to(device).eval()
for p in lpips_net.parameters():
    p.requires_grad_(False)

# ----------------------- 小工具 -----------------------
def safe_to01(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return (x.clamp(-1, 1) * 0.5) + 0.5

def as_bchw(imgs):
    """将 imgs 规范为 [B,C,H,W] float32 Tensor；兼容 (imgs, meta)/list/single-CHW"""
    if isinstance(imgs, (list, tuple)):
        imgs = imgs[0]
    assert isinstance(imgs, torch.Tensor), f"expect Tensor, got {type(imgs)}"
    if imgs.dim() == 3:
        imgs = imgs.unsqueeze(0)
    assert imgs.dim() == 4, f"expect 4D BCHW, got shape {tuple(imgs.shape)}"
    if imgs.dtype != torch.float32:
        imgs = imgs.float()
    return imgs.contiguous()

@torch.no_grad()
def apply_jnd_embed_fre(base_img: torch.Tensor, proposed_img: torch.Tensor, alpha: float) -> torch.Tensor:
    """频域 JND 衰减叠加，输入/输出均为 [-1,1]"""
    with torch.no_grad():
        hmap = jnd_map_fre.heatmaps(safe_to01(base_img)).to(base_img.dtype)
    delta = proposed_img - base_img
    delta = F.avg_pool2d(delta, 3, 1, 1)  # 简单可靠
    return base_img + alpha * hmap * delta

def build_attacks() -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    # 评测用到的一组增强（与你给出的集合保持一致）
    return  {
        'None': lambda x: x,
        'Crop 50%': lambda x: utils_img.center_crop(x, 0.5),
        # 'crop_07': lambda x: utils_img.center_crop(x, 0.7),
        # 'Drop_03': lambda x: utils_img.Drop(x, 0.3),
        'Drop 50%': lambda x: utils_img.Drop(x, 0.5),
        # 'Drop_07': lambda x: utils_img.Drop(x, 0.7),
        'Rot 10%': lambda x: utils_img.rotate(x, 10),
        # 'rot_90': lambda x: utils_img.rotate(x, 90),
        'HorizontalFlip': lambda x: FbyV.hflip(x),
        'Resize 30%': lambda x: utils_img.resize(x, 0.3),
        # 'resize_05': lambda x: utils_img.resize(x, 0.5),
        'Brightness 2.0':   lambda x: utils_img.adjust_brightness_01(x, 2),
        'Hue+0.25':  lambda x: FbyV.adjust_hue(x, 0.25),
        # 'Hue-0.1':  lambda x: FbyV.adjust_hue(x, -0.1),
        'JPEG_50':  lambda x: utils_img.jpeg_compress_01(x, 50),
        # 'jpeg_80':  lambda x: utils_img.jpeg_compress_01(x, 80),
        # 'gaussblur_3':  lambda x: utils_img.gaussian_blur(x, 3),
        'GaussBlur 17': lambda x: utils_img.gaussian_blur(x, 17),
        'GaussNoise 0.05': lambda x: utils_img.GaussianNoise(x, 0.05),
        # 'S&PNoise_0.05':   lambda x: utils_img.sp_noise(x, 0.05),
        'Shear_10': lambda x: utils_img.shear(x, 10),
        'Saturation_2.0': lambda x: utils_img.adjust_saturation_01(x, 2.0),
        # 'median_3': lambda x: utils_img.median_filter(x, 3),
        'MedianFilter 7': lambda x: utils_img.median_filter(x, 7),
        'Perspective 0.5': lambda x: utils_img.adjust_perspective(x, 0.5),
        'Contrast 2.0':    lambda x: utils_img.adjust_contrast_01(x, 2.0),
        'Combination_Attack': lambda x: utils_img.Combination_Attack(x),
    }

# ============== Binomial tail & thresholds (GS Fig.5) ==============
def binom_sf_strict_greater(k: int, tau: int) -> float:
    """P[Acc > tau] for Acc~Binom(k, p=0.5)"""
    total = 0.0
    denom = 2.0 ** k
    for i in range(tau + 1, k + 1):
        total += math.comb(k, i)
    return total / denom

def find_tau_for_fpr(k: int, fpr_target: float) -> int:
    """Detection 阈值：最小 τ 使 P[Acc>τ] <= fpr_target"""
    lo, hi = -1, k
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        f = binom_sf_strict_greater(k, mid)
        if f > fpr_target:
            lo = mid
        else:
            hi = mid
    return hi

def find_tau_for_fpr_with_users(k: int, fpr_target: float, N_users: int) -> int:
    """Trace 阈值：1 - (1 - P_single)^N <= fpr_target"""
    lo, hi = -1, k
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        f_single = binom_sf_strict_greater(k, mid)
        f_multi  = 1.0 - (1.0 - f_single) ** N_users
        if f_multi > fpr_target:
            lo = mid
        else:
            hi = mid
    return hi

# ============== 稳定计算：log10(p-value)（只输出该值和 bit-acc） ==============
@torch.no_grad()
def _binom_tail_log_table(n: int, device: torch.device) -> torch.Tensor:
    """
    预计算尾和表 log_tail[h] = log P[Binom(n,0.5) >= h]  (h=0..n)，float64 精度。
    采用 logcumsumexp，避免极小 p 的下溢。
    """
    # PMF 的 log 形式（natural log）
    ks = torch.arange(0, n + 1, device=device, dtype=torch.float64)
    logC = (torch.lgamma(torch.tensor(n + 1.0, device=device)) -
            torch.lgamma(ks + 1.0) -
            torch.lgamma(torch.tensor(n, device=device) - ks + 1.0))
    logpmf = logC - (n * torch.log(torch.tensor(2.0, device=device, dtype=torch.float64)))
    # 右尾：对从 k 到 n 的 PMF 求和，再取 log —— 用 logcumsumexp 实现
    log_tail = torch.logcumsumexp(logpmf.flip(0), dim=0).flip(0)  # (n+1,)
    return log_tail  # natural log

@torch.no_grad()
def log10p_from_bitacc(bit_accs: torch.Tensor, nbits: int) -> torch.Tensor:
    """
    返回 log10(p-value)，其中 p-value 为右尾 P[X >= k]，k = floor(p*nbits)。
    注意：log10p <= 0；若需要“正的显著性”，可使用 -log10p。
    """
    k = torch.floor(bit_accs * float(nbits) + 1e-12).to(torch.int64).clamp_(0, nbits)
    log_tail = _binom_tail_log_table(int(nbits), device=bit_accs.device)   # [n+1], natural log
    ln10 = torch.log(torch.tensor(10.0, device=bit_accs.device, dtype=log_tail.dtype))
    log10p = (log_tail[k] / ln10).to(torch.float32)  # 转为以 10 为底
    return log10p  # 形状 [B]，非正数

# ----------------------- Parser -----------------------
def _parse_list_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def _parse_list_ints(s: str) -> List[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x: continue
        if "e" in x or "E" in x:
            out.append(int(float(x)))
        else:
            out.append(int(x))
    return out

def get_parser():
    p = argparse.ArgumentParser("Evaluate watermark decoder (val + GS Fig.5 + plots)")
    g = p.add_argument_group("Data")
    g.add_argument("--val_dir", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/data_coco/val2017")
    g.add_argument("--img_size", type=int, default=256)
    g.add_argument("--batch_size", type=int, default=16)
    g.add_argument("--num_imgs", type=int, default=5000)

    g = p.add_argument_group("Models")
    g.add_argument("--ldm_config", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/sd/v2-inference.yaml")
    g.add_argument("--ldm_ckpt", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/sd/v2-1_512-ema-pruned.ckpt")
    g.add_argument("--decoder_ckpt", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/AModel/ori-li0.5+lr0.5-alpha3.0-FRE0.05-MF-NoBlue.pth")
    g.add_argument("--msg_decoder_path", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/videoseal_ori/ckpts/y_256b_img_whit.pth")
    g.add_argument("--num_bits", type=int, default=256)
    g.add_argument("--key_str", type=str, default="1110101101010000010101110100110101000100001001110111110011111111101011100011010111111110100100110001001101001011110100011101111110101000000011010101010001110100111111111001100110011010010010011100001011100010111101110110001100001110110111100000110110011010")
    g.add_argument("--alpha_jnd", type=float, default=3.0)

    g = p.add_argument_group("Augmenter / Metrics / Log")
    g.add_argument("--aug_yaml", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/VS_ckpts_NoW/Image_aug.yaml")
    g.add_argument("--output_dir", type=str, default="AAvalThings/detect&iden-test")
    g.add_argument("--save_img_freq", type=int, default=999999)
    g.add_argument("--log_freq", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)

    # ------ GS Fig.5 Detection / Traceability ------
    g = p.add_argument_group("GS Detection/Traceability")
    g.add_argument("--run_detection", type=utils.bool_inst, default=True)
    g.add_argument("--fpr_grid", type=str, default="1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1")
    g.add_argument("--det_max_batches", type=int, default=0)

    g.add_argument("--run_traceability", type=utils.bool_inst, default=True)
    g.add_argument("--trace_mode", type=str, choices=["analytic","exact"], default="analytic")
    g.add_argument("--N_list", type=str, default="10,100,1000,10000,100000,1000000,10000000")
    g.add_argument("--fpr_target", type=float, default=1e-6)
    g.add_argument("--trace_ckpts", type=str, default="")
    g.add_argument("--trace_keys_file", type=str, default="")
    g.add_argument("--trace_max_batches_per_user", type=int, default=0)
    g.add_argument("--trace_chunk", type=int, default=4096)

    # ------ Plot from json ------
    g = p.add_argument_group("Plots")
    g.add_argument("--make_plots", type=utils.bool_inst, default=True, help="评测后自动绘图")
    g.add_argument("--plot_from_detection_json", type=str, default="", help="仅绘图：读取 detection_curves.json")
    g.add_argument("--plot_from_trace_json", type=str, default="", help="仅绘图：读取 traceability_curves_*.json")

    return p

# ----------------------- val -----------------------
@torch.no_grad()
def val(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module,
        vqgan_to_imnet: Callable, key: torch.Tensor, params: argparse.Namespace, augmenter: Augmenter):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    augmenter.eval()

    # FID：三套
    fid_w_vs_raw  = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_d0_vs_raw = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_w_vs_d0   = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid_w_vs_raw.reset(); fid_d0_vs_raw.reset(); fid_w_vs_d0.reset()

    attacks = build_attacks()

    # 保存前10张（分 10 个子文件夹）
    base_imgsets_dir = Path(params.output_dir) / "imgsets"
    base_imgsets_dir.mkdir(parents=True, exist_ok=True)
    saved_n = 0

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = as_bchw(imgs).to(device)

        # LDM 编解码 + 双 decoder + JND 叠加
        z      = ldm_ae.encode(imgs).mode()
        d0     = ldm_ae.decode(z)
        w_raw  = ldm_decoder.decode(z)
        w      = apply_jnd_embed_fre(d0, w_raw, params.alpha_jnd)

        keys = key.repeat(imgs.shape[0], 1)

        # 基础指标（w vs d0）
        log_stats = {
            "iteration": ii,
            "psnr":  utils_img.psnr(w, d0).mean().item(),
            "ssim":  utils_img.ssim(w, d0).mean().item(),
            "lpips": lpips_net(w, d0).mean().item(),
        }

        # 统一为 [0,1]
        raw_01 = safe_to01(imgs).float().clamp_(0, 1)
        d0_01  = safe_to01(d0).float().clamp_(0, 1)
        w_01   = safe_to01(w).float().clamp_(0, 1)
        wraw_01= safe_to01(w_raw).float().clamp_(0, 1)

        # FID 累计
        fid_w_vs_raw.update(raw_01, real=True); fid_w_vs_raw.update(w_01, real=False)
        fid_d0_vs_raw.update(raw_01, real=True); fid_d0_vs_raw.update(d0_01, real=False)
        fid_w_vs_d0.update(d0_01, real=True);   fid_w_vs_d0.update(w_01,  real=False)

        # —— 对每个增强都计算 bit-acc、log10(p) —— #
        for name, attack in attacks.items():
            imgs_aug = attack(w_01)                 # [0,1]
            logits   = msg_decoder.detect(imgs_aug) # [B,k] logits
            bits     = (logits > 0).float()
            diff     = (~torch.logical_xor(bits > 0, keys > 0))  # [B,k]
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]  # [B]

            # 记录 bit-acc 平均
            log_stats[f'bit_acc_{name}'] = bit_accs.mean().item()

            # 仅输出 log10p（log₁₀ p-value），nbits 来自 shape（通常为 256）
            nbits   = diff.shape[-1]
            log10p  = log10p_from_bitacc(bit_accs, nbits=nbits)     # [B], <= 0
            log_stats[f'log10p_{name}_mean'] = log10p.mean().item()

        # 累计日志
        for name, v in log_stats.items():
            metric_logger.update(**{name: v})

        # —— 保存前 10 张样本：每个样本单独子目录 —— #
        if saved_n < 10:
            B = raw_01.shape[0]
            take = min(B, 10 - saved_n)
            for bi in range(take):
                idx = saved_n + bi
                subdir = base_imgsets_dir / f"sample_{idx:02d}"
                subdir.mkdir(parents=True, exist_ok=True)

                ori  = raw_01[bi:bi+1]
                d0b  = d0_01[bi:bi+1]
                ww   = w_01[bi:bi+1]
                wr   = wraw_01[bi:bi+1]

                # 残差：按 [-1,1] 可视，映射到 [0,1]
                def vis_diff(a01, b01, amp=10.0):
                    dif = ((a01 - b01) * amp).clamp(-1, 1)
                    return (dif + 1.0) / 2.0

                # 顺序与命名
                save_image(ori, subdir / "ori.png")
                save_image(d0b, subdir / "d0.png")
                save_image(wr,  subdir / "w_raw.png")
                save_image(ww,  subdir / "w.png")
                save_image(vis_diff(wr, d0b, 10.0), subdir / "10rawdiff.png")
                save_image(vis_diff(ww, d0b, 10.0), subdir / "10wdiff.png")
                save_image(vis_diff(ww, d0b, 50.0), subdir / "50wdiff.png")
                save_image(vis_diff(ww, ori,  10.0), subdir / "ori10wdiff.png")
                save_image(vis_diff(ww, ori,  50.0), subdir / "ori50wdiff.png")

            saved_n += take

        # 可选保存对齐图（整 batch）
        if ii % params.save_img_freq == 0:
            imgs_dir = Path(params.output_dir) / "imgs"
            imgs_dir.mkdir(parents=True, exist_ok=True)
            save_image(raw_01.clamp(0,1),  imgs_dir / f'{num_eval:03}_val_orig.png', nrow=min(8,raw_01.shape[0]))
            save_image(d0_01.clamp(0,1),   imgs_dir / f'{num_eval:03}_val_d0.png',   nrow=min(8,raw_01.shape[0]))
            save_image(w_01.clamp(0,1),    imgs_dir / f'{num_eval:03}_val_w.png',    nrow=min(8,raw_01.shape[0]))
            save_image(wraw_01.clamp(0,1), imgs_dir / f'{num_eval:03}_val_w_raw.png',nrow=min(8,raw_01.shape[0]))

    # 汇总 FID
    out_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    out_stats["fid_w_vs_raw"]  = fid_w_vs_raw.compute().item()
    out_stats["fid_d0_vs_raw"] = fid_d0_vs_raw.compute().item()
    out_stats["fid_w_vs_d0"]   = fid_w_vs_d0.compute().item()

    print(f"[VAL] FID(w,raw)={out_stats['fid_w_vs_raw']:.6f} | "
          f"FID(d0,raw)={out_stats['fid_d0_vs_raw']:.6f} | "
          f"FID(w,d0)={out_stats['fid_w_vs_d0']:.6f}")
    print("Averaged eval stats:", out_stats)
    return out_stats

# ----------------------- 生成 w(0..1) 批次（供 GS 评测重复使用） -----------------------
@torch.no_grad()
def _forward_w_batches(
    data_loader: Iterable[torch.Tensor],
    ldm_ae: nn.Module,
    ldm_decoder: nn.Module,
    alpha_jnd: float,
    max_batches: Optional[int]=None
) -> List[torch.Tensor]:
    out = []
    count = 0
    for imgs in data_loader:
        imgs = as_bchw(imgs).to(device)
        z   = ldm_ae.encode(imgs).mode()
        d0  = ldm_ae.decode(z)
        w_r = ldm_decoder.decode(z)
        w   = apply_jnd_embed_fre(d0, w_r, alpha_jnd)   # [-1,1]
        out.append(safe_to01(w))                        # [0,1]
        count += 1
        if max_batches and count >= max_batches:
            break
    return out

# ----------------------- GS: Detection -----------------------
@torch.no_grad()
def eval_detection(
    data_loader: Iterable[torch.Tensor],
    ldm_ae: nn.Module,
    ldm_decoder: nn.Module,
    msg_decoder: nn.Module,
    key_bits: torch.Tensor,               # [k]
    alpha_jnd: float,
    fpr_grid: List[float],
    attacks: Dict[str,Callable],
    det_max_batches: int = 0
) -> Dict[str, List[Dict[str,float]]]:
    k = int(key_bits.numel())
    key_bits = key_bits.to(device).float().unsqueeze(0)

    w_batches = _forward_w_batches(
        data_loader, ldm_ae, ldm_decoder, alpha_jnd,
        None if det_max_batches<=0 else det_max_batches
    )

    results: Dict[str, List[Dict[str,float]]] = {name: [] for name in attacks.keys()}

    for name, atk in attacks.items():
        all_acc: List[int] = []
        for w01 in w_batches:
            logits = msg_decoder.detect(atk(w01).to(device))
            bits   = (logits > 0).float()
            keys   = key_bits.repeat(bits.shape[0], 1)
            acc    = (bits.eq(keys)).sum(dim=1).cpu().tolist()
            all_acc.extend(acc)
        # 确保 FPR 从小到大
        fpr_sorted = sorted(fpr_grid)
        for fpr in fpr_sorted:
            tau = find_tau_for_fpr(k, fpr)
            tpr = sum(1.0 if a > tau else 0.0 for a in all_acc) / max(1, len(all_acc))
            results[name].append({"fpr": float(fpr), "tpr": float(tpr), "tau": int(tau)})
    return results

# ----------------------- GS: Traceability -----------------------
@torch.no_grad()
def eval_traceability(
    data_loaders_per_user: List[Iterable[torch.Tensor]],
    ldm_ae: nn.Module,
    ldm_decoders: List[nn.Module],
    user_keys: List[torch.Tensor],          # 每个 [k]
    msg_decoder: nn.Module,
    alpha_jnd: float,
    N_list: List[int],
    fpr_target: float,
    attacks: Dict[str,Callable],
    mode: str = "analytic",
    max_batches_per_user: int = 0,
    chunk: int = 4096
) -> Dict[str, List[Dict[str,float]]]:
    assert mode in {"analytic","exact"}
    U = len(ldm_decoders)
    assert U == len(user_keys) == len(data_loaders_per_user)
    k = int(user_keys[0].numel())
    user_keys = [uk.to(device).float().view(1,k) for uk in user_keys]

    # 预生成每个用户的 w01 批次
    w_per_user: List[List[torch.Tensor]] = []
    for u in range(U):
        wlist = _forward_w_batches(
            data_loaders_per_user[u], ldm_ae, ldm_decoders[u], alpha_jnd,
            None if max_batches_per_user<=0 else max_batches_per_user
        )
        w_per_user.append(wlist)

    results: Dict[str, List[Dict[str,float]]] = {name: [] for name in attacks.keys()}

    for name, atk in attacks.items():
        # 收集“真用户匹配位数”分布
        all_acc_true: List[int] = []
        for u in range(U):
            key_true = user_keys[u]
            for w01 in w_per_user[u]:
                logits = msg_decoder.detect(atk(w01).to(device))
                bits   = (logits > 0).float()
                acc    = (bits.eq(key_true.repeat(bits.shape[0],1))).sum(dim=1).cpu().tolist()
                all_acc_true.extend(acc)

        for N in N_list:
            tau = find_tau_for_fpr_with_users(k, fpr_target, N)

            if mode == "analytic":
                correct_prob_sum = 0.0
                count = 0
                for a in all_acc_true:
                    if a <= tau:
                        count += 1
                        continue
                    p_gt = binom_sf_strict_greater(k, a)       # P[Acc > a]
                    p_ok = (1.0 - p_gt) ** (N - 1)
                    correct_prob_sum += p_ok
                    count += 1
                acc_est = correct_prob_sum / max(1, count)
                results[name].append({"N": int(N), "acc": float(acc_est), "tau": int(tau)})
            else:
                # exact：构造 N 个 key（若 N>U，补随机）
                base_keys = torch.cat(user_keys, dim=0)  # [U,k]
                if N > U:
                    extra = torch.randint(0, 2, (N-U, k), device=device, dtype=torch.float32)
                    all_keys = torch.cat([base_keys, extra], dim=0)  # [N,k]
                else:
                    all_keys = base_keys[:N]

                total, correct = 0, 0
                for u in range(U):
                    key_true = user_keys[u]
                    for w01 in w_per_user[u]:
                        logits = msg_decoder.detect(atk(w01).to(device))   # [B,k]
                        bits   = (logits > 0).float()                      # [B,k]
                        acc_true = (bits.eq(key_true)).sum(dim=1)          # [B]
                        detected = (acc_true > tau)
                        if not torch.any(detected):
                            continue

                        # 分块与 N 个 key 做匹配
                        B = bits.shape[0]
                        best_idx = torch.full((B,), -1, device=device, dtype=torch.long)
                        best_acc = torch.full((B,), -1_0000, device=device, dtype=torch.long)
                        for s in range(0, all_keys.shape[0], chunk):
                            e = min(s+chunk, all_keys.shape[0])
                            kblk = all_keys[s:e].to(torch.bool)           # [M,k]
                            b_exp = bits.to(torch.bool).unsqueeze(1)      # [B,1,k]
                            k_exp = kblk.unsqueeze(0)                      # [1,M,k]
                            acc_blk = (b_exp.eq(k_exp)).sum(dim=2)        # [B,M]
                            cur_acc, cur_idx = torch.max(acc_blk, dim=1)  # [B]
                            take = cur_acc > best_acc
                            best_acc[take] = cur_acc[take]
                            best_idx[take] = (cur_idx[take] + s)

                        if N <= u:
                            is_correct = torch.zeros_like(detected, dtype=torch.bool, device=device)
                        else:
                            is_correct = best_idx.eq(u)
                        correct += int(is_correct[detected].sum().item())
                        total   += int(detected.sum().item())

                acc_exact = (correct / total) if total > 0 else 0.0
                results[name].append({"N": int(N), "acc": float(acc_exact), "tau": int(tau)})

    return results

# ----------------------- 绘图（读取 JSON 或直接接收 dict） -----------------------
def _dynamic_attack_styles(names: List[str]):
    """
    根据传入的攻击名称列表生成颜色/marker 样式映射，长度可扩展。
    """
    # 一长串易区分的颜色（Tableau + 其他）
    colors = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#393b79","#637939","#8c6d31","#843c39","#7b4173",
        "#3182bd","#31a354","#756bb1","#e6550d","#636363",
        "#9c9ede","#8ca252","#bd9e39","#ad494a","#ce6dbd",
        "#6baed6","#74c476","#9e9ac8","#fd8d3c","#bdbdbd"
    ]
    markers = ['o','s','^','v','D','*','P','X','>','<','h','H','d','p','|','_']
    style = {}
    for i, n in enumerate(names):
        style[n] = dict(color=colors[i % len(colors)], marker=markers[i % len(markers)])
    return style

# ---- 论文风格：全局/坐标轴设定（纯白背景、无格线） ----
def _set_paper_rc():
    import matplotlib as mpl
    plt.style.use("default")
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
    })

def _apply_paper_axes(ax):
    ax.grid(False)
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.0); sp.set_color("black")
    ax.tick_params(axis="both", which="both", colors="black",
                   direction="out", length=3.0, width=1.0)
    ax.set_axisbelow(False)

def _save_vec_and_png(fig, out_path: Path, dpi=600):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)

def plot_detection_curves(curves: Dict[str, List[Dict[str,float]]], out_png: Path, title=None):
    _set_paper_rc()
    # 使用 curves 的 key 顺序（即 build_attacks 的定义顺序）
    order = list(curves.keys())
    styles = _dynamic_attack_styles(order)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ymin = 1.0
    for name in order:
        pts = sorted(curves[name], key=lambda d: float(d["fpr"]))
        xs = [float(d["fpr"]) for d in pts]
        ys = [float(d["tpr"]) for d in pts]
        if ys: ymin = min(ymin, min(ys))
        ax.plot(xs, ys, lw=1.8, ms=4, label=name, **styles[name])
    ax.set_xscale("log")
    ax.set_ylim(bottom=max(0.0, ymin - 0.01), top=1.0)
    ax.margins(y=0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    _apply_paper_axes(ax)
    # 根据曲线数量自动决定列数
    ncol = 2 if len(order) <= 12 else 3
    ax.legend(ncol=ncol, fontsize=9)
    if title:
        ax.set_title(title, pad=6)
    fig.tight_layout()
    _save_vec_and_png(fig, out_png)

def plot_traceability_curves(curves: Dict[str, List[Dict[str,float]]], out_png: Path, title=None):
    _set_paper_rc()
    order = list(curves.keys())
    styles = _dynamic_attack_styles(order)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name in order:
        pts = sorted(curves[name], key=lambda d: int(d["N"]))
        xs = [int(d["N"]) for d in pts]
        ys = [float(d["acc"]) for d in pts]
        ax.plot(xs, ys, lw=1.8, ms=4, label=name, **styles[name])
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.margins(y=0)
    ax.set_xlabel("Number of Users")
    ax.set_ylabel("Accuracy of Traceability")
    _apply_paper_axes(ax)
    ncol = 2 if len(order) <= 12 else 3
    ax.legend(ncol=ncol, fontsize=9)
    if title:
        ax.set_title(title, pad=6)
    fig.tight_layout()
    _save_vec_and_png(fig, out_png)

def plot_detection_from_json(json_path: str, out_png: Optional[str]=None):
    curves = json.loads(Path(json_path).read_text())
    out_png = out_png or (Path(json_path).with_suffix(".png"))
    plot_detection_curves(curves, Path(out_png))

def plot_traceability_from_json(json_path: str, out_png: Optional[str]=None):
    curves = json.loads(Path(json_path).read_text())
    out_png = out_png or (Path(json_path).with_suffix(".png"))
    plot_traceability_curves(curves, Path(out_png))

# ----------------------- 主流程：加载 + val + GS + plot -----------------------
def main(args: argparse.Namespace):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 仅绘图（读取现有 JSON）模式
    if args.plot_from_detection_json or args.plot_from_trace_json:
        if args.plot_from_detection_json:
            plot_detection_from_json(args.plot_from_detection_json,
                                     out_png=str(Path(args.plot_from_detection_json).with_suffix(".png")))
            print(f"[PLOT] detection figure saved to {Path(args.plot_from_detection_json).with_suffix('.png')}")
        if args.plot_from_trace_json:
            plot_traceability_from_json(args.plot_from_trace_json,
                                        out_png=str(Path(args.plot_from_trace_json).with_suffix(".png")))
            print(f"[PLOT] traceability figure saved to {Path(args.plot_from_trace_json).with_suffix('.png')}")
        return

    # 1) LDM AE
    print(f'>>> Building LDM AE from {args.ldm_config} & {args.ldm_ckpt}')
    config = OmegaConf.load(args.ldm_config)
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, args.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    ldm_ae.to(device).eval()

    # 2) 载入我们的水印 decoder
    print(f'>>> Loading watermark decoder from {args.decoder_ckpt}')
    ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    dec_sd = ckpt.get("ldm_decoder", ckpt)
    ldm_decoder = deepcopy(ldm_ae)
    ldm_decoder.encoder = nn.Identity()
    ldm_decoder.quant_conv = nn.Identity()
    ldm_decoder.load_state_dict(dec_sd, strict=False)
    ldm_decoder.to(device).eval()
    for p in ldm_decoder.parameters():
        p.requires_grad_(False)

    # 3) videoseal 检测器
    print(f'>>> Loading msg decoder (videoseal) from {args.msg_decoder_path}')
    msg_decoder = torch.jit.load(args.msg_decoder_path, map_location="cpu").to(device).eval()

    # 4) dataloader
    vqgan_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    val_loader = utils.get_dataloader(args.val_dir, vqgan_transform,
                                      batch_size=args.batch_size,
                                      num_imgs=args.num_imgs, shuffle=False, num_workers=4)

    # 5) augmenter（yaml 与训练一致）
    with open(args.aug_yaml, 'r', encoding='utf-8') as f:
        aug_cfg = yaml.load(f, Loader=yaml.FullLoader)
    augmenter = Augmenter(**aug_cfg).to(device).eval()

    # 6) key
    if args.key_str:
        assert set(args.key_str) <= {'0', '1'}
        assert len(args.key_str) == args.num_bits
        key_bits = torch.tensor([int(b) for b in args.key_str], dtype=torch.float32, device=device).unsqueeze(0)
        print(f'>>> Using provided key ({len(args.key_str)} bits)')
    else:
        key_bits = torch.randint(0, 2, (1, args.num_bits), dtype=torch.float32, device=device)
        print('>>> Using RANDOM key (for quick sanity check)')

    # 7) val
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    base_stats = val(val_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key_bits, args, augmenter)
    (out_dir / "val_metrics.json").write_text(json.dumps(base_stats, ensure_ascii=False, indent=2))

    # ==================== GS Fig.5：Detection ====================
    attacks = build_attacks()
    if args.run_detection:
        print(">>> Running GS-Detection curves ...")
        fpr_grid = _parse_list_floats(args.fpr_grid)
        det_curves = eval_detection(
            val_loader, ldm_ae, ldm_decoder, msg_decoder,
            key_bits.squeeze(0), args.alpha_jnd,
            fpr_grid, attacks,
            det_max_batches=args.det_max_batches
        )
        det_json = out_dir / "detection_curves.json"
        det_json.write_text(json.dumps(det_curves, ensure_ascii=False, indent=2))
        print(f"[GS-Detection] saved {det_json}")

        if args.make_plots:
            plot_detection_curves(det_curves, out_dir / "detection_curves")
            print(f"[PLOT] detection figure saved to {out_dir/'detection_curves.(png|pdf|svg)'}")

    # ==================== GS Fig.5：Traceability ====================
    if args.run_traceability:
        print(f">>> Running GS-Traceability ({args.trace_mode}) ...")
        ckpt_list = [p for p in args.trace_ckpts.split(",") if p.strip()] if args.trace_ckpts else []
        decoders_list: List[nn.Module] = []
        keys_list: List[torch.Tensor] = []

        if ckpt_list:
            for path in ckpt_list:
                ckpt_u = torch.load(path, map_location="cpu")
                dec_sd_u = ckpt_u.get("ldm_decoder", ckpt_u)
                dec_u = deepcopy(ldm_ae)
                dec_u.encoder = nn.Identity(); dec_u.quant_conv = nn.Identity()
                dec_u.load_state_dict(dec_sd_u, strict=False)
                dec_u.to(device).eval()
                decoders_list.append(dec_u)
        else:
            decoders_list = [ldm_decoder]  # 单用户也可跑 analytic

        if args.trace_keys_file and os.path.isfile(args.trace_keys_file):
            lines = [ln.strip() for ln in Path(args.trace_keys_file).read_text().splitlines() if ln.strip()]
            for ln in lines:
                assert set(ln) <= {'0','1'} and len(ln)==args.num_bits
                keys_list.append(torch.tensor([int(b) for b in ln], dtype=torch.float32, device=device))
        else:
            if args.key_str:
                keys_list = [key_bits.squeeze(0) for _ in range(len(decoders_list))]
            else:
                for _ in range(len(decoders_list)):
                    keys_list.append(torch.randint(0,2,(args.num_bits,), device=device, dtype=torch.float32))

        dls = [val_loader for _ in range(len(decoders_list))]
        N_list = _parse_list_ints(args.N_list)
        trace_curves = eval_traceability(
            dls, ldm_ae, decoders_list, keys_list, msg_decoder,
            args.alpha_jnd, N_list, args.fpr_target, attacks,
            mode=args.trace_mode,
            max_batches_per_user=args.trace_max_batches_per_user,
            chunk=args.trace_chunk
        )
        out_name = f"traceability_curves_{args.trace_mode}.json"
        (out_dir / out_name).write_text(json.dumps(trace_curves, ensure_ascii=False, indent=2))
        print(f"[GS-Traceability] saved {out_name}")

        if args.make_plots:
            plot_traceability_curves(trace_curves, out_dir / f"traceability_curves_{args.trace_mode}")
            print(f"[PLOT] traceability figure saved to {out_dir/f'traceability_curves_{args.trace_mode}.(png|pdf|svg)'}")

# ----------------------- 入口 -----------------------
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
