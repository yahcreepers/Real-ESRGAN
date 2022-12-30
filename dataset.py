import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional_tensor import rgb_to_grayscale
from torchvision.utils import save_image
from scipy import special
import cv2
from torch.nn import functional as F

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img

def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D sinc filter
    Reference: https://dsp.stackexchange.com/questions/58301/2-d-circularly-symmetric-low-pass-filter
    Args:
        cutoff (float): cutoff frequency in radians (pi is max)
        kernel_size (int): horizontal and vertical size, must be odd.
        pad_to (int): pad kernel size to desired size, must be odd or zero.
    """
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    kernel = np.fromfunction(
        lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)) / (2 * np.pi * np.sqrt(
                (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2)), [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff**2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel

def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)

def sinc(hr):
    kernels = []
    for _ in range(hr.shape[0]):
        kernel_size = random.choice([2 * i + 1 for i in range(3, 11)])
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        pad_size = (21 - kernel_size) // 2
        kernel = torch.FloatTensor(np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))).to(hr.device)
        kernels.append(kernel)
    kernels = torch.stack(kernels)
    img = filter2D(hr, kernels)
    return img
    

def gaussian_noise(hr, sigma_range, gray_prob):
    sigma = torch.rand(hr.shape[0], dtype=hr.dtype, device=hr.device).view(hr.shape[0], 1, 1, 1) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    noise = torch.randn(*hr.shape, dtype=hr.dtype, device=hr.device) * sigma / 255.
    gray_noise = torch.randn(hr.shape[0], dtype=hr.dtype, device=hr.device)
    gray_noise = (gray_noise < gray_prob).float().view(hr.shape[0], 1, 1, 1)
    noise_gray = torch.randn(*hr.shape[2:4], dtype=hr.dtype, device=hr.device) * sigma / 255.
    noise_gray = noise_gray.view(hr.shape[0], 1, hr.shape[2], hr.shape[3])
    noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    hr = noise + hr
    hr = torch.clamp(hr, 0, 1)
    return hr

def poisson_noise(hr, scale_range, gray_prob):
    gray_noise = torch.randn(hr.shape[0], dtype=hr.dtype, device=hr.device)
    gray_noise = (gray_noise < gray_prob).float().view(hr.shape[0], 1, 1, 1)
    hr_gray = torch.clamp((hr * 255.0).round(), 0, 255) / 255.
    vals_list = [len(torch.unique(hr_gray[i, :, :, :])) for i in range(hr.shape[0])]
    vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
    vals = hr_gray.new_tensor(vals_list).view(hr.shape[0], 1, 1, 1)
    out = torch.poisson(hr_gray * vals) / vals
    noise_gray = out - hr_gray
    noise_gray = noise_gray.expand(hr.shape[0], 3, hr.shape[2], hr.shape[3])
    
    hr = torch.clamp((hr * 255.0).round(), 0, 255) / 255.
    vals_list = [len(torch.unique(hr[i, :, :, :])) for i in range(hr.shape[0])]
    vals_list = [2**np.ceil(np.log2(vals)) for vals in vals_list]
    vals = hr.new_tensor(vals_list).view(hr.shape[0], 1, 1, 1)
    out = torch.poisson(hr * vals) / vals
    noise = out - hr
    noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    scale = torch.rand(hr.shape[0], dtype=hr.dtype, device=hr.device).view(hr.shape[0], 1, 1, 1) * (scale_range[1] - scale_range[0]) + scale_range[0]
    noise = noise * scale
    hr = noise + hr
    hr = torch.clamp(hr, 0, 1)
    return hr
conunt = 0
def high_order_degradation(hr, jpeger, sinc_prob1=0.1, sinc_prob2=0.1, resize_prob1=[0.15, 1.5], resize_prob2=[0.3, 1.2], gaussian_noise_prob1=0.5, gaussian_noise_prob2=0.5, sigma_range1=[1, 30], sigma_range2=[1, 25], scale_range1=[0.05, 3], scale_range2=[0.05, 2.5], gray_prob1=0.4, gray_prob2=0.4, jpeg_range1=[30, 95], jpeg_range2=[30, 95]):
    ori_h, ori_w = hr.shape[2:4]
    global conunt
    #Blur
    if np.random.uniform() < sinc_prob1:
        hr = sinc(hr)
    else:
        kernel = random.choice([2 * i + 1 for i in range(3, 11)])
        hr = transforms.functional.gaussian_blur(hr, kernel)
    #save_image(hr, f'./img/high_order_imm/{conunt}_blur.png')
    
    #Resize
    scale = np.random.uniform(resize_prob1[0], resize_prob1[1])
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    hr = F.interpolate(hr, scale_factor=scale, mode=mode)
    #save_image(hr, f'./img/high_order_imm/{conunt}_resize.png')
    
    #Add Noise
    if np.random.uniform() < gaussian_noise_prob1:
        hr = gaussian_noise(hr, sigma_range1, gray_prob1)
    else:
        hr = poisson_noise(hr, scale_range1, gray_prob1)
    #save_image(hr, f'./img/high_order_imm/{conunt}_noise.png')
    
    #JPEG
    jpeg_p = hr.new_zeros(hr.shape[0]).uniform_(*jpeg_range1)
    hr = jpeger(hr, quality=jpeg_p)
    #save_image(hr, f'./img/high_order_imm/{conunt}_jpeg.png')
    
    #Blur
    kernel = random.choice([2 * i + 1 for i in range(3, 11)])
    if np.random.uniform() < sinc_prob1:
        hr = sinc(hr)
    else:
        kernel = random.choice([2 * i + 1 for i in range(3, 11)])
        hr = transforms.functional.gaussian_blur(hr, kernel)
    #save_image(hr, f'./img/high_order_imm/{conunt}_blur2.png')
    
    #Resize
    scale = np.random.uniform(resize_prob2[0], resize_prob2[1])
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    hr = F.interpolate(hr, scale_factor=scale, mode=mode)
    #save_image(hr, f'./img/high_order_imm/{conunt}_resize2.png')
    
    #Add Noise
    if np.random.uniform() < gaussian_noise_prob2:
        hr = gaussian_noise(hr, sigma_range2, gray_prob2)
    else:
        hr = poisson_noise(hr, scale_range2, gray_prob2)
    #save_image(hr, f'./img/high_order_imm/{conunt}_noise2.png')
    
    #JPEG
    kernel = random.choice([2 * i + 1 for i in range(3, 11)])
    if np.random.uniform() < 0.5:
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        hr = F.interpolate(hr, size=(ori_h // 4, ori_w // 4), mode=mode)
        hr = sinc(hr)
        jpeg_p = hr.new_zeros(hr.shape[0]).uniform_(*jpeg_range2)
        hr = jpeger(hr, quality=jpeg_p)
    else:
        jpeg_p = hr.new_zeros(hr.shape[0]).uniform_(*jpeg_range2)
        hr = jpeger(hr, quality=jpeg_p)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        hr = F.interpolate(hr, size=(ori_h // 4, ori_w // 4), mode=mode)
        hr = sinc(hr)
    #save_image(hr, f'./img/high_order/{conunt}_high_order.png')
    conunt += 1
    return hr

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape):
        
        self.random = transforms.Compose(
            [
                transforms.RandomCrop(size=((hr_shape, hr_shape))),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.lowres = transforms.Compose(
            [
                transforms.Resize((hr_shape // 4, hr_shape // 4), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.highres = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_r = self.random(img)
        img_lr = self.lowres(img_r)
        img_hr = self.highres(img_r)

        return img_lr, img_hr

    def __len__(self):
        return len(self.files)

class Real_Dataset(Dataset):
    def __init__(self, root, hr_shape):
        self.random = transforms.Compose(
            [
                transforms.RandomCrop(size=((hr_shape, hr_shape)), pad_if_needed=True, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        self.small = transforms.Compose(
                [
                    #transforms.Resize((hr_shape, hr_shape), Image.BICUBIC),
                    transforms.ToTensor(),
                ]
        )
        self.files = sorted(glob.glob(root + "/*.*"))
    
    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = img.convert('RGB')
        img = self.random(img)
        #img = self.small(img)
        return img
    
    def __len__(self):
        return len(self.files)

class TestDataset(Dataset):
    def __init__(self, root, hr_shape):
        self.trans = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
        )
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = img.convert('RGB')
        img = self.trans(img)
        return img

    def __len__(self):
        return len(self.files)
#jpeger = DiffJPEG(differentiable=False)
#a = torch.randn((5, 3, 512, 512))
#b = high_order_degradation(a, jpeger)
#print(a.shape, b.shape)
