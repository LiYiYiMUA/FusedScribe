# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyright: reportMissingModuleSource=false

import numpy as np
from augly.image import functional as aug_functional
import torch
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

from PIL import Image, ImageFilter
import pytorch_msssim
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

normalize_vqgan = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize (x - 0.5) / 0.5             [0,1]->[-1,1]
unnormalize_vqgan = transforms.Normalize(mean=[-1, -1, -1], std=[1/0.5, 1/0.5, 1/0.5]) # Unnormalize (x * 0.5) + 0.5      [-1,1]->[0,1]
normalize_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize (x - mean) / std   [0,1]->imagenet    
unnormalize_img = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]) # Unnormalize (x * std) + mean   将 ImageNet 标准化图像还原为 [0,1] 范围

def psnr(x, y, img_space='vqgan'):
    """ 
    Return PSNR 
    Args:
        x: Image tensor with values approx. between [-1,1]
        y: Image tensor with values approx. between [-1,1], ex: original image
    """
    if img_space == 'vqgan':
        delta = torch.clamp(unnormalize_vqgan(x), 0, 1) - torch.clamp(unnormalize_vqgan(y), 0, 1)
    elif img_space == 'img':
        delta = torch.clamp(unnormalize_img(x), 0, 1) - torch.clamp(unnormalize_img(y), 0, 1)
    else:
        delta = x - y
    delta = 255 * delta
    delta = delta.reshape(-1, x.shape[-3], x.shape[-2], x.shape[-1]) # BxCxHxW
    psnr = 20*np.log10(255) - 10*torch.log10(torch.mean(delta**2, dim=(1,2,3)))  # B
    return psnr

# ─── SSIM（结构相似性）──────────────────────────────────────────
def ssim(x: torch.Tensor,
         y: torch.Tensor,
         img_space: str = "vqgan",
         data_range: float = 1.0) -> torch.Tensor:
    """
    Return SSIM (Structural Similarity Index)

    Args:
        x: Image tensor             (≈ [-1,1] in 'vqgan'/'img' spaces)
        y: Reference image tensor   (same format as x)
        img_space: 'vqgan' | 'img' | 'raw'
                   - 'vqgan' : call unnormalize_vqgan(), expect [-1,1] → [0,1]
                   - 'img'   : call unnormalize_img(),  expect [-1,1] → [0,1]
                   - other   : assume already in [0,1],直接参与计算
        data_range: pixel dynamic range fed to pytorch_msssim (默认 1.0)
    Returns:
        Tensor of shape (B,) with SSIM values in [0,1]
    """
    # —— 1. 把输入规范到 [0,1] ————————————————————————————
    if img_space == "vqgan":
        x_01 = torch.clamp(unnormalize_vqgan(x), 0, 1)
        y_01 = torch.clamp(unnormalize_vqgan(y), 0, 1)
    else:   # 已经是 [0,1]
        x_01, y_01 = x, y

    # —— 2. 计算 SSIM ————————————————————————————————
    # size_average=False ⇒ 返回逐样本向量 (B,)
    return pytorch_msssim.ssim(
        x_01, y_01,
        data_range=data_range,
        size_average=False
    )

def center_crop(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.center_crop(x, new_edges_size)

def center_crop_pil(x, scale):
    """
    Perform center crop on a PIL image such that the target area 
    is at the given scale (e.g., 0.5 means cropping to 50% area).
    
    Args:
        x (PIL.Image.Image): Input PIL image.
        scale (float): Target area scale, e.g., 0.5 for 50%.
    
    Returns:
        PIL.Image.Image: Cropped image.
    """
    scale = np.sqrt(scale)
    width, height = x.size  # PIL image size is (width, height)
    new_height = int(height * scale)
    new_width = int(width * scale)
    return F.center_crop(x, [new_height, new_width])


# 添加Drop
def Drop(x: torch.Tensor, drop_ratio: float) -> torch.Tensor:
    """
    在图像中随机遮罩一块矩形区域，输入输出均为 Tensor。

    参数:
        x (torch.Tensor): 输入图像，形状为 (C,H,W) 或 (B,C,H,W)，像素值 ∈ [0,1]
        drop_ratio (float): 遮罩比例（0~1）

    返回:
        torch.Tensor: 与输入形状一致的图像，遮罩区域为 0
    """
    single = x.ndim == 3  # (C,H,W)
    if single:
        x = x.unsqueeze(0)  # 转为 (1,C,H,W)

    B, C, H, W = x.shape
    out = x.clone()

    new_height = max(1, int(H * drop_ratio))
    new_width = max(1, int(W * drop_ratio))

    for i in range(B):
        top = torch.randint(0, H - new_height + 1, (1,)).item()
        left = torch.randint(0, W - new_width + 1, (1,)).item()
        out[i, :, top:top + new_height, left:left + new_width] = 0.0  # 遮罩区域设为 0

    if single:
        out = out.squeeze(0)
    return out


def resize(x, scale):
    """ Perform center crop such that the target area of the crop is at a given scale
    Args:
        x: PIL image
        scale: target area scale 
    """
    scale = np.sqrt(scale)
    new_edges_size = [int(s*scale) for s in x.shape[-2:]][::-1]
    return functional.resize(x, new_edges_size)

def rotate(x, angle):
    """ Rotate image by angle
    Args:
        x: image (PIl or tensor)
        angle: angle in degrees
    """
    return functional.rotate(x, angle)

def adjust_brightness(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return normalize_img(functional.adjust_brightness(unnormalize_img(x), brightness_factor))

# to 01
def adjust_brightness_01(x, brightness_factor):
    """ Adjust brightness of an image
    Args:
        x: PIL image
        brightness_factor: brightness factor
    """
    return (functional.adjust_brightness((x), brightness_factor))

def adjust_contrast(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return normalize_img(functional.adjust_contrast(unnormalize_img(x), contrast_factor))

#  to 01
def adjust_contrast_01(x, contrast_factor):
    """ Adjust contrast of an image
    Args:
        x: PIL image
        contrast_factor: contrast factor
    """
    return (functional.adjust_contrast((x), contrast_factor))

def adjust_saturation(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return normalize_img(functional.adjust_saturation(unnormalize_img(x), saturation_factor))

# to 01
def adjust_saturation_01(x, saturation_factor):
    """ Adjust saturation of an image
    Args:
        x: PIL image
        saturation_factor: saturation factor
    """
    return (functional.adjust_saturation((x), saturation_factor))

def adjust_hue(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return normalize_img(functional.adjust_hue(unnormalize_img(x), hue_factor))

# to 01
def adjust_hue_01(x, hue_factor):
    """ Adjust hue of an image
    Args:
        x: PIL image
        hue_factor: hue factor
    """
    return (functional.adjust_hue((x), hue_factor))

def adjust_gamma(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return normalize_img(functional.adjust_gamma(unnormalize_img(x), gamma, gain))

#  to 01
def adjust_gamma_01(x, gamma, gain=1):
    """ Adjust gamma of an image
    Args:
        x: PIL image
        gamma: gamma factor
        gain: gain factor
    """
    return (functional.adjust_gamma((x), gamma, gain))

def adjust_sharpness(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return normalize_img(functional.adjust_sharpness(unnormalize_img(x), sharpness_factor))

# to 01
def adjust_sharpness_01(x, sharpness_factor):
    """ Adjust sharpness of an image
    Args:
        x: PIL image
        sharpness_factor: sharpness factor
    """
    return (functional.adjust_sharpness((x), sharpness_factor))

def overlay_text(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text))
    return normalize_img(img_aug)

# to 01
def overlay_text_01(x, text='Lorem Ipsum'):
    """ Overlay text on image
    Args:
        x: PIL image
        text: text to overlay
        font_path: path to font
        font_size: font size
        color: text color
        position: text position
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)

    # 将文本字符串转换为Unicode码点列表（支持多行）
    lines = text.split('\n')
    text_list = [[ord(c) for c in line] for line in lines]

    for ii,img in enumerate(x):
        pil_img = to_pil((img))
        img_aug[ii] = to_tensor(aug_functional.overlay_text(pil_img, text=text_list))
    return (img_aug)

def jpeg_compress(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil(unnormalize_img(img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return normalize_img(img_aug)

# to 01
def jpeg_compress_01(x, quality_factor):
    """ Apply jpeg compression to image
    Args:
        x: PIL image
        quality_factor: quality factor
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    img_aug = torch.zeros_like(x, device=x.device)
    for ii,img in enumerate(x):
        pil_img = to_pil((img))
        img_aug[ii] = to_tensor(aug_functional.encoding_quality(pil_img, quality=quality_factor))
    return (img_aug)


def gaussian_blur(x, ksize):
    return torchvision.transforms.functional.gaussian_blur(x, ksize)

def shear(x, degrees):
    return torchvision.transforms.functional.affine(x, angle=0,
                                                    translate=[0,0],
                                                    scale=1.0,
                                                    shear=[degrees,0])

def get_perspective_params(width, height, distortion_scale):
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale *
                half_width) + 1, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale *
                half_height) + 1, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale *
                half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale *
                half_height) + 1, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale *
                half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale *
                half_height) - 1, height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale *
                half_width) + 1, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale *
                half_height) - 1, height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0],
                       [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

#  to 01
def adjust_perspective(
    x,
    radio: float = 0.5,
    interpolation: str = 'bilinear',
    fillcolor: int = 0,
):
    """Apply a 0.5-strength perspective transform to a normalized tensor.
    Args:
        x:             normalized image tensor (e.g. from normalize_img)
        interpolation: 'nearest' or 'bilinear'
        fillcolor:     pixel fill value for areas outside the transformed image
    """
    # 1. 字符串转枚举
    if isinstance(interpolation, str):
        mode = interpolation.lower()
        if mode == 'bilinear':
            interp_mode = InterpolationMode.BILINEAR
        elif mode == 'nearest':
            interp_mode = InterpolationMode.NEAREST
        else:
            raise ValueError(f"Unsupported interpolation mode: {interpolation}")
    elif isinstance(interpolation, InterpolationMode):
        interp_mode = interpolation
    else:
        raise TypeError("interpolation must be a str or InterpolationMode")

    # 2. 
    pil_img = (x)  # PIL.Image

    # 3. 计算 0.5 强度的透视参数
    width, height = x.shape[-1], x.shape[-2]
    startpoints, endpoints = get_perspective_params(width, height, radio)

    # 4. 调用 torchvision.functional.perspective
    transformed_pil = F.perspective(
        pil_img,
        startpoints=startpoints,
        endpoints=endpoints,
        interpolation=interp_mode,
        fill=fillcolor
    )

    # 5. 归一化回张量并返回
    return (transformed_pil)

def sp_noise(x: torch.Tensor, prob: float) -> torch.Tensor:
    """
    Apply salt-and-pepper noise to a batch of images.

    Args:
        x (torch.Tensor): Input image(s), shape (B, C, H, W) or (C, H, W), values in [0,1].
        prob (float): Total probability of noise (salt + pepper), must be in [0,1].

    Returns:
        torch.Tensor: Noised image(s), same shape and device, values in [0,1].
    """
    if not (0.0 <= prob <= 1.0):
        raise ValueError("prob must be between 0 and 1")

    # 支持 (C,H,W) 或 (B,C,H,W)
    single = (x.ndim == 3)
    if single:
        x = x.unsqueeze(0)  # 转为 (1,C,H,W)

    # salt & pepper 概率
    prob_zero = prob / 2.0
    prob_one = 1.0 - prob_zero

    # 生成随机矩阵
    noise = torch.rand_like(x)

    # 深拷贝输入
    out = x.clone()

    # pepper: 随机置 0
    out = torch.where(noise > prob_one, torch.zeros_like(out), out)
    # salt:  随机置 1
    out = torch.where(noise < prob_zero, torch.ones_like(out), out)

    if single:
        out = out.squeeze(0)
    return out

def median_filter(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply a median filter to a batch of images.

    Parameters:
        images (torch.Tensor): The input images tensor of shape BxCxHxW.
        kernel_size (int): The size of the median filter kernel.

    Returns:
        torch.Tensor: The filtered images.
    """
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    # Compute the padding size
    padding = kernel_size // 2
    # Pad the images
    images_padded = torch.nn.functional.pad(
        images, (padding, padding, padding, padding))
    # Extract local blocks from the images
    blocks = images_padded.unfold(2, kernel_size, 1).unfold(
        3, kernel_size, 1)  # BxCxHxWxKxK
    # Compute the median of each block
    medians = blocks.median(dim=-1).values.median(dim=-1).values  # BxCxHxW
    return medians

def md_f(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    对一个 batch 的图像 Tensor (B,C,H,W) 应用 PIL 的中值滤波。
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    B = images.shape[0]
    out = torch.zeros_like(images)

    for i in range(B):
        pil_img = to_pil(images[i].cpu())
        pil_filtered = pil_img.filter(ImageFilter.MedianFilter(kernel_size))
        out[i] = to_tensor(pil_filtered).to(images.device)

    return out


# def median_filter(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
#     """
#     Apply a median filter to a batch of images.

#     Parameters:
#         images (torch.Tensor): The input images tensor of shape BxCxHxW.
#         kernel_size (int): The size of the median filter kernel (must be odd).

#     Returns:
#         torch.Tensor: The filtered images.
#     """
#     # Check input
#     if not isinstance(images, torch.Tensor):
#         raise TypeError("Input must be a PyTorch Tensor")
#     if images.dim() != 4:
#         raise ValueError("Input must be 4D tensor of shape [B, C, H, W]")
#     if kernel_size % 2 == 0:
#         raise ValueError("Kernel size must be odd.")

#     # Handle NaN values
#     if torch.isnan(images).any():
#         print("NaN detected in median_filter input")
#         images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)

#     # Compute padding size
#     padding = kernel_size // 2

#     # Pad images with reflect mode
#     images_padded = torch.nn.functional.pad(
#         images, (padding, padding, padding, padding))

#     # Extract local blocks
#     blocks = images_padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # BxCxHxWxKxK
#     blocks = blocks.reshape(*blocks.shape[:-2], -1)  # BxCxHxWx(K*K)

#     # Compute median
#     medians = blocks.median(dim=-1).values  # BxCxHxW

#     # Ensure output range
#     medians = torch.clamp(medians, 0, 1)

#     # Handle NaN in output
#     if torch.isnan(medians).any():
#         print("NaN detected in median_filter output")
#         medians = torch.nan_to_num(medians, nan=0.0, posinf=1.0, neginf=0.0)

#     return medians

import torch

def GaussianNoise(image, std=None):
    """
    向图像添加高斯噪声。

    参数:
    - image: 输入图像张量，形状为 (B, C, H, W)，数据类型为 float。
    - std: 噪声的标准差（控制噪声强度）。若为 None，则随机生成标准差。

    返回:
    - 添加噪声后的图像张量，与输入图像形状相同。
    """
    # 如果未指定 std，则生成随机标准差（例如从 [0.01, 0.2] 范围内随机选择）
    if std is None:
        std = torch.rand(1) * 0.09 + 0.01  # 随机 std ∈ [0.01, 0.1]

    # 生成与图像形状相同的高斯噪声
    noise = torch.randn_like(image) * std

    # 将噪声添加到图像中
    noisy_image = image + noise

    # 裁剪像素值到 [0, 1] 范围（假设输入图像已归一化）
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)

    return noisy_image


def Combination_Attack(x: torch.Tensor) -> torch.Tensor:
    """
    Apply a combination attack: JPEG-50, brightness 2.0, and center-crop 70%
    on a batch of image tensors in [0,1], using tensor operations.

    Args:
        x (torch.Tensor): Tensor image batch, shape (B, C, H, W), values in [0,1].

    Returns:
        torch.Tensor: Attacked tensor images, same shape, values in [0,1].
    """
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    out = torch.zeros_like(x, device=x.device)
    B = x.shape[0]

    for i in range(B):
        # tensor → PIL for JPEG
        pil = to_pil(x[i].cpu())
        pil = aug_functional.encoding_quality(pil, quality=80)
        img_tensor = to_tensor(pil).to(x.device)  # back to tensor in [0,1]

        # adjust brightness in tensor domain
        img_tensor = F.adjust_brightness(img_tensor, brightness_factor=1.5)

        # center crop with tensor-based version
        img_tensor = center_crop(img_tensor, scale=0.5)

        # resize back to original size
        img_tensor = F.resize(img_tensor, size=[x.shape[2], x.shape[3]])

        out[i] = img_tensor

    return out



