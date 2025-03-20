torchrun --nproc_per_node=1 \
main.py \
--config test.json \
--do_test \
--epochs=20 \
--binary_threshold 0.25 \
--gpu_id '0' \
--seed 1 \
--aligned \
> test_memor1.log 2>&1 