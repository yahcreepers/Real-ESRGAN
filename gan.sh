CUDA_VISIBLE_DEVICES=$1 python train.py \
	--do_train \
	--real \
	--batch_size 8 \
	--accumulate 6 \
	--step 2400000 \
	--eval_epoch 10 \
	--lr 1e-4 \
	--model_path test/pretrain/generator.pt #epoch=49-PSNR=22.67.ckpt
