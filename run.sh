# torchrun --standalone --nproc_per_node=8 train_gpt.py
# torchrun --standalone --nproc_per_node=1 train_single_gpu.py \
#     --gpt.flex_kernel_consumer True --train.sequence_length 32768 --train.batch_size 32
torchrun --standalone --nproc_per_node=1 train_mango.py \
    --gpt.flex_kernel_consumer True --train.sequence_length 32768 --train.batch_size 32
