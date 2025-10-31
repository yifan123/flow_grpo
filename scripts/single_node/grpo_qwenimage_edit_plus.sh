# 2 GPU
torchrun --standalone --nproc_per_node=2 --master_port=19507 scripts/train_qwenimage_edit_plus.py --config config/grpo.py:dual_ref_qwenimage_edit_plus_fast