import torch
import numpy as np
import os
import copy
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from model import *
from module import *
import torchvision.transforms as transforms
from PIL import Image
from dataset import ImageDataset, Real_Dataset, TestDataset, rgb2ycbcr_pt
import random


def main(args):

    if args.do_predict:
        dataset_name = TestDataset
    elif args.real:
        dataset_name = Real_Dataset
    else:
        dataset_name = ImageDataset

    if args.real:
        if isinstance(args.train_data_dir, list):
            trainset = dataset_name(args.train_data_dir[0], args.hr_size)
            for i in range(1, len(args.train_data_dir)):
                trainset += dataset_name(args.train_data_dir[i], args.hr_size)
        if isinstance(args.train_data_dir, str):
            trainset = dataset_name(args.train_data_dir, args.hr_size)
        validset = dataset_name(args.valid_data_dir, args.hr_size)
        generator = RRDBNet(3, 3, 64, 23, gc=32)
        if (args.do_train or args.do_pretrain or args.do_predict) and not args.checkpoint:
            generator.load_state_dict(torch.load(args.model_path))
        #discriminator = Discriminator((3, args.hr_size, args.hr_size))
        discriminator = UNetDiscriminatorSN(num_in_ch=3)
    else:
        if isinstance(args.train_data_dir, list):
            trainset = dataset_name(args.train_data_dir[0], args.hr_size)
            for i in range(1, len(args.train_data_dir)):
                trainset += dataset_name(args.train_data_dir[i], args.hr_size)
        if isinstance(args.train_data_dir, str):
            trainset = dataset_name(args.train_data_dir, args.hr_size)
        validset = dataset_name(args.valid_data_dir, args.hr_size)
        generator = Hyper_RRDBNet()
        discriminator = Discriminator((3, args.hr_size, args.hr_size))
        
        
    extractor = FeatureExtractor()
    
#    psnr = torchmetrics.PeakSignalNoiseRatio()
#    count = 0
#    generator = RRDBNet(3, 3, 64, 23, gc=32)
#    generator.load_state_dict(torch.load(args.model_path))
#    validloader = DataLoader(validset, batch_size=args.batch_size)
#    for i, batch in enumerate(validloader):
#        lr, hr = batch
#        gen_hr = generator(lr)
#        gen_hr = torch.clamp(gen_hr, 0, 1)
#        gen_yc = rgb2ycbcr_pt(gen_hr, True)
#        hr_yc = rgb2ycbcr_pt(hr, True)
#        P = psnr(gen_yc, hr_yc)
#        print(P)
#    exit(0)
    
    model_kwargs = {
        "generator": generator,
        "discriminator": discriminator,
        "extractor": extractor,
        "real": args.real,
        "batch_size": args.batch_size,
        "trainset": trainset,
        "validset": validset,
        "do_pretrain": args.do_pretrain,
        "lr": args.pretrain_lr if args.do_pretrain else args.lr,
        "output_dir": f"{args.output_dir}",
        "accumulate": args.accumulate,
        "load_lr": args.load_lr,
    }
    model = ESRGAN(**model_kwargs)
#    model.load_state_dict(torch.load(args.model_path)["state_dict"])
#    torch.save(model.G.state_dict(), f"{args.output_dir}/pretrain/generator.pt")
#    exit(0)
    callbacks = []
    name = "pretrain" if args.do_pretrain else "GANtrain"
    
    if args.do_train:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{args.output_dir}/{name}",
            filename="{epoch}-{PSNR:.2f}",
            save_last=True,
            save_weights_only=True,
            every_n_epochs=args.eval_epoch,
        )
        callbacks.append(checkpoint_callback)
    if args.do_pretrain:
        checkpoint_callback = ModelCheckpoint(
            monitor="PSNR",
            dirpath=f"{args.output_dir}/{name}",
            filename="{epoch}-{PSNR:.2f}",
            save_top_k=5,
            mode="max",
            every_n_epochs=args.eval_epoch,
            save_weights_only=True,
        )
        callbacks.append(checkpoint_callback)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{args.output_dir}/{name}")
    trainer_kwargs = {
        "max_epochs": args.epoch,
        "max_steps": args.pretrain_steps if args.do_pretrain else args.steps,
        "accelerator": "gpu",
        "check_val_every_n_epoch": args.eval_epoch,
        "gpus": args.cuda,
        "logger": tb_logger,
        "deterministic": False,
        "callbacks": callbacks,
    }
    
    if args.do_pretrain:
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model)
        torch.save(model.G.state_dict(), f"{args.output_dir}/{name}/last_generator.pt")
        model.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)["state_dict"])
        torch.save(model.G.state_dict(), f"{args.output_dir}/{name}/generator.pt")
        #generator.load_state_dict(torch.load(f"{args.output_dir}/{name}/gernerator.pt"))
    if args.do_train:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.model_path)["state_dict"])
        if args.do_pretrain:
            model.lr = args.lr
            tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"{args.output_dir}/GANtrain")
            trainer_kwargs["logger"] = tb_logger
            trainer_kwargs["max_epochs"] = args.epoch
#        temp = ESRGAN.load_state_dict(torch.load(args.model_path)["state_dict"])
#        model.G.load_state_dict(temp.G.state_dict())
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(model)
    
    if args.do_predict:
        #if not args.do_train:
        #    model.load_state_dict(torch.load(args.model_path)["state_dict"])
        #    torch.save(model.G.state_dict(), f"model/Real-ESRGAN-v2.pt")
        #    exit(0)
        trainer = pl.Trainer(**trainer_kwargs)
        trainer.test(model)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model/RRDB_ESRGAN_x4.pth")
    parser.add_argument("--train_data_dir", default=["data/DIV2K_train_HR/", "data/Flickr2K", "data/OST"])
    parser.add_argument("--valid_data_dir", default="data/DIV2K_valid_HR/")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--early_stop", type=int, default=None)
    parser.add_argument("--eval_epoch", type=int, default=10)
    parser.add_argument("--pretrain_steps", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hr_size", type=int, default=400)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--do_pretrain", action="store_true")
    parser.add_argument("--pretrain_epoch", type=int, default=40000)
    parser.add_argument("--pretrain_lr", type=float, default=2e-4)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--load_lr", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
