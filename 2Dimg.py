# -*- coding: utf-8 -*-
"""
Generate watermarked images and residual visualizations for a folder of images,
and compute PSNR/SSIM/LPIPS metrics.

Outputs per input image:
  - watermarked/<name>_w.png         and watermarked/<name>_w.pdf
  - imgs_raw/<name>_raw.png          and imgs_raw/<name>_raw.pdf        <-- NEW: decoder 初步输出
  - residual/<name>_res.png          and residual/<name>_res.pdf
  - residual_x10/<name>_res10.png    and residual_x10/<name>_res10.pdf
    * NOTE: res10 is absolute residual * 10, i.e., clamp(|w01 - raw01| * 10, 0, 1)
Metrics:
  - metrics.csv (per-image rows + final 'AVG' row)
  - metrics_avg.json
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as TVF
from PIL import Image
import lpips

# --- 项目内依赖 ---
import utils
import utils_img
import utils_model

sys.path.append('src')
from omegaconf import OmegaConf
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion

# JND
from jndWAM import JND
from Jnd_fre import JND as JND_fre
# ===== 用 JNDen 替换原先的 jndWAM/Jnd_fre =====
from jndEn import JND as JNDen

# ----------------------- 全局 -----------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 固定导出分辨率（仅影响保存，不影响指标计算）
EXPORT_SIZE = 512

# 单例：JND（与你给定一致：频域 + blue=True）
jnd_map_fre = JND_fre(in_channels=1, out_channels=3, blue=True, freq=False, w_fm=0.05).to(device).eval()
for p in jnd_map_fre.parameters():
    p.requires_grad_(False)

# 单例：LPIPS (alex)
lpips_net = lpips.LPIPS(net='alex').to(device).eval()
for p in lpips_net.parameters():
    p.requires_grad_(False)


def safe_to01(x: torch.Tensor) -> torch.Tensor:
    """把[-1,1]域张量映射到[0,1]（用于保存/可视化）；对超界值做裁剪。"""
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return (x.clamp(-1, 1) * 0.5) + 0.5


@torch.no_grad()
def apply_jnd_embed_fre(base_img: torch.Tensor, proposed_img: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    频域 JND 衰减叠加：输入/输出均为 [-1,1]
      base_img    : 参照图（d0）
      proposed_img: 水印decoder原始输出 w_raw
    额外对 delta 做 3x3 平滑以更稳健。
    """
    with torch.no_grad():
        hmap = jnd_map_fre.heatmaps(safe_to01(base_img)).to(base_img.dtype)
    delta = proposed_img - base_img
    # delta = F.avg_pool2d(delta, 3, 1, 1)  # 轻量平滑
    return base_img + alpha * hmap * delta

# ===== JNDen 单例 & 融合函数 =====
jnd_en = JNDen(in_channels=1, out_channels=3, blue=True).to(device).eval()  # 与你原代码保持 blue=True
for p in jnd_en.parameters():
    p.requires_grad_(False)

def apply_jnd_embed_en(base_img: torch.Tensor, proposed_img: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    x_w = x_base + α · JNDen(x_base) · (x_prop - x_base)
    输入/输出均为 [-1,1]
    """
    with torch.no_grad():
        hmap = jnd_en.heatmaps(safe_to01(base_img)).to(base_img.dtype)
    delta = proposed_img - base_img
    return base_img + alpha * hmap * delta

# ----------------------- 数据集 -----------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class FlatImageFolder(torch.utils.data.Dataset):
    def __init__(self, root: str, transform, limit: int = None):
        self.paths: List[Path] = sorted(
            [p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS]
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)  # [-1,1]
        return x, str(p)


# ----------------------- 模型加载 -----------------------
def build_models(ldm_config: str, ldm_ckpt: str, decoder_ckpt: str) -> Tuple[AutoencoderKL, AutoencoderKL]:
    """加载 LDM AE 与你的水印 decoder（仅用其 decoder 部分）。"""
    print(f"[Load] LDM from {ldm_config} & {ldm_ckpt}")
    config = OmegaConf.load(ldm_config)
    ldm: LatentDiffusion = utils_model.load_model_from_config(config, ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm.first_stage_model
    ldm_ae.to(device).eval()
    for p in ldm_ae.parameters():
        p.requires_grad_(False)

    print(f"[Load] Watermark decoder ckpt: {decoder_ckpt}")
    ckpt = torch.load(decoder_ckpt, map_location="cpu")
    dec_sd = ckpt.get("ldm_decoder", ckpt)  # 兼容字段或整包

    # 克隆 AE，只保留 decoder 部分参数（与评测一致）
    ldm_decoder = deepcopy(ldm_ae)
    ldm_decoder.encoder = nn.Identity()
    ldm_decoder.quant_conv = nn.Identity()
    missing, unexpected = ldm_decoder.load_state_dict(dec_sd, strict=False)
    if missing or unexpected:
        print(f"[Warn] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
    ldm_decoder.to(device).eval()
    for p in ldm_decoder.parameters():
        p.requires_grad_(False)

    return ldm_ae, ldm_decoder


# ----------------------- 保存工具（固定导出为 512x512 + PDF） -----------------------
def _resize_to_512(x01: torch.Tensor) -> torch.Tensor:
    """x01: [B,3,H,W] in [0,1] -> resize to [B,3,512,512] by bilinear."""
    if x01.dim() != 4:
        raise ValueError("Expected x01 with shape [B,3,H,W]")
    return F.interpolate(x01, size=(EXPORT_SIZE, EXPORT_SIZE), mode="bilinear", align_corners=False)


def _save_png_and_pdf(t01: torch.Tensor, file_png: Path, file_pdf: Path):
    """
    t01: [3,H,W] in [0,1] (CPU)
    保存 PNG（torchvision） + PDF（PIL）
    """
    save_image(t01.unsqueeze(0), file_png)
    pil_img = TVF.to_pil_image(t01)  # [3,H,W] in [0,1] -> PIL RGB
    pil_img.save(file_pdf, format="PDF")


def save_triplet(
    imgs_w: torch.Tensor, imgs_raw_for_residual: torch.Tensor, in_paths: List[str],
    out_w_dir: Path, out_res_dir: Path, out_res10_dir: Path,
    fmt: str = "png",
    # NEW: 额外导出 decoder 初步输出
    imgs_raw_decoder: torch.Tensor = None,
    out_imgs_raw_dir: Path = None,
):
    """
    保存水印图、残差图、**10×残差绝对值**三件套（各自 PNG + PDF）。
    残差定义：res = w - raw_for_residual   （raw_for_residual 为原图在预处理后的张量）

    另外（NEW）：
      若提供 imgs_raw_decoder（即 w_raw），则将其导出到 out_imgs_raw_dir 目录下。
    """
    assert imgs_w.shape == imgs_raw_for_residual.shape
    B = imgs_w.shape[0]

    # 映射 & 上采样
    w01_512     = _resize_to_512(safe_to01(imgs_w).cpu())                   # [B,3,512,512]
    raw01_512   = _resize_to_512(safe_to01(imgs_raw_for_residual).cpu())    # [B,3,512,512]
    wraw01_512  = None
    if imgs_raw_decoder is not None:
        assert imgs_raw_decoder.shape == imgs_w.shape
        wraw01_512 = _resize_to_512(safe_to01(imgs_raw_decoder).cpu())

    # 残差（带符号）：在 [-1,1] 计算 -> [0,1] 可视化 -> 上采样
    res_signed = imgs_w - imgs_raw_for_residual
    res01_512  = _resize_to_512(safe_to01(res_signed).cpu())

    # 十倍残差绝对值（在 [0,1] 域做差）
    res10abs01_512 = (w01_512 - raw01_512).abs() * 10.0
    res10abs01_512 = res10abs01_512.clamp_(0.0, 1.0)

    for i in range(B):
        stem = Path(in_paths[i]).stem

        # 路径
        path_w_png   = out_w_dir       / f"{stem}_w.{fmt}"
        path_w_pdf   = out_w_dir       / f"{stem}_w.pdf"
        path_r_png   = out_res_dir     / f"{stem}_res.{fmt}"
        path_r_pdf   = out_res_dir     / f"{stem}_res.pdf"
        path_r10_png = out_res10_dir   / f"{stem}_res10.{fmt}"
        path_r10_pdf = out_res10_dir   / f"{stem}_res10.pdf"

        # 逐张保存（PNG + PDF）
        _save_png_and_pdf(w01_512[i],        path_w_png,   path_w_pdf)
        _save_png_and_pdf(res01_512[i],      path_r_png,   path_r_pdf)
        _save_png_and_pdf(res10abs01_512[i], path_r10_png, path_r10_pdf)

        # NEW: 保存 decoder 初步输出 imgs_raw（即 w_raw）
        if wraw01_512 is not None and out_imgs_raw_dir is not None:
            raw_png = out_imgs_raw_dir / f"{stem}_raw.{fmt}"
            raw_pdf = out_imgs_raw_dir / f"{stem}_raw.pdf"
            _save_png_and_pdf(wraw01_512[i], raw_png, raw_pdf)


# ----------------------- 指标工具 -----------------------
def tensor_to_scalar_list(x: torch.Tensor) -> List[float]:
    """把形如 [B] 或 [B,1,...] 的张量转为 Python 浮点列表。"""
    x = x.detach().flatten().tolist()
    return [float(v) for v in x]


def compute_batch_metrics(imgs_a: torch.Tensor, imgs_b: torch.Tensor):
    """
    计算一批图像的 PSNR/SSIM/LPIPS（逐样本），输入均在 [-1,1] 域。
    返回：{'psnr': [B], 'ssim': [B], 'lpips': [B]}
    """
    psnr_vals = utils_img.psnr(imgs_a, imgs_b)  # [B]
    ssim_vals = utils_img.ssim(imgs_a, imgs_b)  # [B]
    lpips_vals = lpips_net(imgs_a, imgs_b)      # [B,1,1,1] or [B]
    return {
        'psnr':  tensor_to_scalar_list(psnr_vals),
        'ssim':  tensor_to_scalar_list(ssim_vals),
        'lpips': tensor_to_scalar_list(lpips_vals),
    }


# ----------------------- CLI -----------------------
def get_parser():
    p = argparse.ArgumentParser("Batch-generate watermarked images, residual maps, and metrics (PNG+PDF export)")
    p.add_argument("--input_dir", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/AAthings/AAreal-gen", help="输入图片文件夹（递归）")
    p.add_argument("--output_dir", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/AAthings/AAreal-gen-dis", help="输出根目录")
    p.add_argument("--ldm_config", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/sd/v2-inference.yaml")
    p.add_argument("--ldm_ckpt", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/sd/v2-1_512-ema-pruned.ckpt")
    p.add_argument("--decoder_ckpt", type=str, default="/mnt/nfs/liyi/liyi_data/FT_ldm/stable_signature/AAASNEWtrain/ori-enJND-oldLoss-ALLAUG-15w-3-ONLYwvg+ONLYprior/checkpoint_stage2_jnden.pth")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--alpha_jnd", type=float, default=2.5, help="JND 频域衰减强度")
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 张，调试用")
    p.add_argument("--format", type=str, default="png", choices=["png", "jpg", "jpeg", "webp"])
    p.add_argument("--amp", action="store_true", help="CUDA 上使用 autocast 加速")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metrics_csv", type=str, default="metrics.csv")
    p.add_argument("--metrics_json", type=str, default="metrics_avg.json")
    return p


# ----------------------- 主流程 -----------------------
@torch.inference_mode()
def main():
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 输出目录
    out_root = Path(args.output_dir)
    out_w_dir = out_root / "watermarked"
    out_imgs_raw_dir = out_root / "imgs_raw"        # NEW: decoder 初步输出目录
    out_res_dir = out_root / "residual"
    out_res10_dir = out_root / "residual_x10"
    for d in [out_root, out_w_dir, out_imgs_raw_dir, out_res_dir, out_res10_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 指标 CSV
    csv_path = out_root / args.metrics_csv
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    header = [
        "filename",
        # w vs raw（主参照）
        "psnr_w_raw", "ssim_w_raw", "lpips_w_raw",
        # w vs d0（嵌入扰动量）
        "psnr_w_d0",  "ssim_w_d0",  "lpips_w_d0",
        # d0 vs raw（重建质量）
        "psnr_d0_raw","ssim_d0_raw","lpips_d0_raw",
    ]
    writer.writerow(header)

    # 模型
    ldm_ae, ldm_decoder = build_models(args.ldm_config, args.ldm_ckpt, args.decoder_ckpt)

    # 变换：与 VQGAN/LDM 一致的归一化（输出到 [-1,1]）
    tfm = transforms.Compose([
        transforms.Resize(args.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,  # -> [-1,1]
    ])

    # 数据
    ds = FlatImageFolder(args.input_dir, tfm, limit=args.limit)
    if len(ds) == 0:
        print(f"[Error] No images found in: {args.input_dir}")
        csv_file.close()
        return
    print(f"[Info] Found {len(ds)} images")
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda")
    )

    # AMP 环境（可选）
    amp_ctx = torch.cuda.amp.autocast if (device.type == "cuda" and args.amp) else contextlib.nullcontext

    # 全局累计（用于平均）
    sums = {
        "psnr_w_raw": 0.0, "ssim_w_raw": 0.0, "lpips_w_raw": 0.0,
        "psnr_w_d0":  0.0, "ssim_w_d0":  0.0, "lpips_w_d0":  0.0,
        "psnr_d0_raw":0.0, "ssim_d0_raw":0.0, "lpips_d0_raw":0.0,
    }
    count = 0

    # 推理
    for step, batch in enumerate(dl):
        imgs, paths = batch          # imgs: [-1,1] 预处理后的原图
        imgs = imgs.to(device, non_blocking=True)

        with amp_ctx():
            # 编解码
            z = ldm_ae.encode(imgs).mode()   # latent
            d0 = ldm_ae.decode(z)            # 原始解码器输出（JND基准）
            w_raw = ldm_decoder.decode(z)    # 水印解码器原始输出（未JND）
            w = apply_jnd_embed_en(d0, w_raw, args.alpha_jnd)  # JND频域衰减

        # 保存三件套 + NEW: imgs_raw（decoder 初步输出）
        save_triplet(
            imgs_w=w,
            imgs_raw_for_residual=imgs,
            in_paths=list(paths),
            out_w_dir=out_w_dir,
            out_res_dir=out_res_dir,
            out_res10_dir=out_res10_dir,
            fmt=args.format,
            imgs_raw_decoder=w_raw,                 # NEW
            out_imgs_raw_dir=out_imgs_raw_dir       # NEW
        )

        # ---- 计算指标（逐样本） ----
        # 1) w vs raw
        M_wr = compute_batch_metrics(w, imgs)
        # 2) w vs d0
        M_wd0 = compute_batch_metrics(w, d0)
        # 3) d0 vs raw
        M_d0r = compute_batch_metrics(d0, imgs)

        B = len(paths)
        for i in range(B):
            stem = Path(paths[i]).name  # 带扩展名，便于回溯
            row = [
                stem,
                M_wr["psnr"][i],  M_wr["ssim"][i],  M_wr["lpips"][i],
                M_wd0["psnr"][i], M_wd0["ssim"][i], M_wd0["lpips"][i],
                M_d0r["psnr"][i], M_d0r["ssim"][i], M_d0r["lpips"][i],
            ]
            writer.writerow(row)

            # 累计
            sums["psnr_w_raw"]  += float(M_wr["psnr"][i])
            sums["ssim_w_raw"]  += float(M_wr["ssim"][i])
            sums["lpips_w_raw"] += float(M_wr["lpips"][i])

            sums["psnr_w_d0"]   += float(M_wd0["psnr"][i])
            sums["ssim_w_d0"]   += float(M_wd0["ssim"][i])
            sums["lpips_w_d0"]  += float(M_wd0["lpips"][i])

            sums["psnr_d0_raw"] += float(M_d0r["psnr"][i])
            sums["ssim_d0_raw"] += float(M_d0r["ssim"][i])
            sums["lpips_d0_raw"]+= float(M_d0r["lpips"][i])

        count += B

        if step % 20 == 0:
            print(f"[{step:04d}] saved batch of {B} | CSV rows: {count}")

    # ---- 写入均值 ----
    if count > 0:
        avgs = {k: v / count for k, v in sums.items()}
        writer.writerow([
            "AVG",
            avgs["psnr_w_raw"],  avgs["ssim_w_raw"],  avgs["lpips_w_raw"],
            avgs["psnr_w_d0"],   avgs["ssim_w_d0"],   avgs["lpips_w_d0"],
            avgs["psnr_d0_raw"], avgs["ssim_d0_raw"], avgs["lpips_d0_raw"],
        ])
        # 保存 JSON
        avg_json = out_root / args.metrics_json
        with open(avg_json, "w", encoding="utf-8") as f:
            json.dump(avgs, f, ensure_ascii=False, indent=2)
        print(f"[Averages]")
        for k, v in avgs.items():
            print(f"  {k}: {v:.6f}")
        print(f"[Done] Saved averages to {avg_json}")

    csv_file.close()
    print(f"[Done] Per-image metrics saved to {csv_path}")
    print(f"[Done] All images saved under {out_root}")


if __name__ == "__main__":
    main()
