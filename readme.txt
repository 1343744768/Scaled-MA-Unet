------Single GPU--------
python train.py

------DDP-------
torch<1.10
python -m torch.distributed.launch --nproc_per_node=number_GPUs train.py
torch>=1.10
python -m torch.distributed.run --nproc_per_node=number_GPUs train.py

# If the memory of GPU is not released after training, use
pgrep python | xargs kill -s 9
