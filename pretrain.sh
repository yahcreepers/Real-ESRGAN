CUDA_VISIBLE_DEVICES=$1 python train.py --do_pretrain \
	--real \
	--batch_size 12 \
	--accumulate 4 \
	--pretrain_steps 4000000 \
	--eval_epoch 10 \
	--lr 2e-4 \
	--model_path model/generator2.pt #test/pretrain/epoch=0-PSNR=21.81.ckpt
