# 1 GPU
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3_gardo.py --config config/grpo_gardo.py:general_ocr_sd3

