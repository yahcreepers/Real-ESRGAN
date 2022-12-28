import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision
from torch.utils.data import DataLoader, random_split
from diffjpeg import DiffJPEG
import torchvision.transforms as transforms
import torchmetrics
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from dataset import denormalize, rgb2ycbcr_pt, high_order_degradation
from torchvision.utils import make_grid, save_image

class ESRGAN(pl.LightningModule):
    def __init__(self, generator, discriminator, extractor, batch_size, lr, trainset, validset, real, do_pretrain, output_dir, gan_w=0.1, l1_w=1.0, accumulate=1):
        super().__init__()
        self.G = generator
        self.D = discriminator
        self.E = extractor
        self.jpeger = DiffJPEG(differentiable=False)
        self.batch_size = batch_size
        self.lr = lr
        self.train_dataset = trainset
        self.valid_dataset = validset
        self.gan_w = gan_w
        self.l1_w = l1_w
        self.real = real
        self.do_pretrain = do_pretrain
        self.output_dir = output_dir
        self.valstep = 0
        self.teststep = 0
        self.accumulate = accumulate
        self.per_weights = [0.1, 0.1, 1.0, 1.0, 1.0]
        
        self.L1loss = torch.nn.L1Loss()
        self.GANloss = torch.nn.BCEWithLogitsLoss()
        self.psnr = torchmetrics.PeakSignalNoiseRatio()
        
        self.automatic_optimization = False

    def configure_optimizers(self):
        lr = self.lr
        g_opt = torch.optim.Adam(self.G.parameters(), lr=lr)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=lr)
        return g_opt, d_opt
        
    def training_step(self, batch, batch_idx):
        if self.real:
            real_hr = batch
            lr = high_order_degradation(real_hr, self.jpeger)
        else:
            lr, real_hr = batch
        
        g_opt, d_opt = self.optimizers()
        ones = torch.ones((lr.shape[0], *self.D.output_shape), requires_grad=True).to(lr.device)
        zeros = torch.zeros((lr.shape[0], *self.D.output_shape), requires_grad=True).to(lr.device)
        
        gen_hr = self.G(lr)
        #gen_hr = torch.clamp(gen_hr, 0, 1)
        
        loss_l1 = self.L1loss(gen_hr, real_hr)
        if self.do_pretrain:
            self.manual_backward(loss_l1)
            if (batch_idx + 1) % self.accumulate == 0:
                self.log_dict({"loss_l1": loss_l1}, on_step=True, prog_bar=True, logger=True)
                g_opt.step()
                g_opt.zero_grad()
            return
        
        real_img = self.D(real_hr).detach()
        fake_img = self.D(gen_hr).detach()
        loss_gan = self.GANloss(fake_img - real_img.mean(0, keepdim=True), ones)
        
        real_feature = self.E(real_hr)
        fake_feature = self.E(gen_hr)
        loss_content = 0
        for i in range(len(real_feature)):
            loss_content += self.L1loss(fake_feature[i], real_feature[i]) * self.per_weights[i]
        
        loss_G = self.l1_w * loss_l1 + self.gan_w * loss_gan + loss_content
        
        self.manual_backward(loss_G)
        if (batch_idx + 1) % self.accumulate == 0:
            g_opt.step()
            g_opt.zero_grad()
        
        loss_real = self.GANloss(real_img - fake_img.mean(0, keepdim=True), ones)
        loss_fake = self.GANloss(fake_img - real_img.mean(0, keepdim=True), zeros)
        
        loss_D = (loss_real + loss_fake) / 2
        
        
        self.manual_backward(loss_D)
        if (batch_idx + 1) % self.accumulate == 0:
            d_opt.step()
            d_opt.zero_grad()
            self.log_dict({"loss_G": loss_G, "loss_D": loss_D}, on_step=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        if self.real:
            hr = batch
            #self.jpeger = self.jpeger.to(real_hr.device)
            lr = high_order_degradation(hr, self.jpeger)
        else:
            lr, hr = batch
        
        gen_hr = self.G(lr)
        gen_hr = torch.clamp(gen_hr, 0, 1)
        gen_yc = rgb2ycbcr_pt(gen_hr, True)
        hr_yc = rgb2ycbcr_pt(hr, True)
        psnr = self.psnr(gen_yc, hr_yc)
        if self.valstep % 100 == 0:
            trans = torchvision.transforms.Resize((hr.shape[-2], hr.shape[-1]))
            nums = min(self.batch_size, 10)
            lrs = trans(lr[:nums])
            images = torch.concat((lrs, gen_hr[:nums], hr[:nums]), 0)
            if not self.real:
                images = denormalize(images)
            grid = make_grid(images, nrow=nums)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'valid_images_{self.current_epoch}_{self.valstep}]', grid, self.valstep)
        self.valstep += 1
        
        return {"PSNR": psnr}
    
    def validation_epoch_end(self, outputs):
        avg_PSNR = torch.stack([x["PSNR"] for x in outputs]).mean()
        self.log("PSNR", avg_PSNR, prog_bar=True, logger=True)
        self.valstep = 0
        return {"PSNR": avg_PSNR}
    
    def test_step(self, batch, batch_idx):
        if self.real:
            hr = batch
            #self.jpeger = self.jpeger.to(real_hr.device)
            lr = high_order_degradation(hr, self.jpeger)
        else:
            lr, hr = batch
        
        gen_hr = self.G(lr)
        gen_hr = torch.clamp(gen_hr, 0, 1)
        gen_yc = rgb2ycbcr_pt(gen_hr, True)
        hr_yc = rgb2ycbcr_pt(hr, True)
        psnr = self.psnr(gen_yc, hr_yc)
        trans = torchvision.transforms.Resize((hr.shape[-2], hr.shape[-1]))
        nums = min(self.batch_size, 10)
        lrs = trans(lr[:nums])
        images = torch.concat((lrs, gen_hr[:nums], hr[:nums]), 0)
        if not self.real:
            images = denormalize(images)
        grid = make_grid(images, nrow=nums)
        save_image(grid, f"{self.output_dir}/{self.teststep}.png")
        self.teststep += 1
        
        return {"PSNR": psnr}
    
    def test_epoch_end(self, outputs):
        avg_PSNR = torch.stack([x["PSNR"] for x in outputs]).mean()
        self.log("PSNR", avg_PSNR, prog_bar=True, logger=True)
        return {"PSNR": avg_PSNR}
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
