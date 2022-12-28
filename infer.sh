source /tmp2/b09902008/miniconda3/etc/profile.d/conda.sh
conda activate ESRGAN
CUDA_VISIBLE_DEVICES=$1 python train.py --real --do_predict --model_path test/GANtrain/epoch\=9-PSNR\=18.90.ckpt 
