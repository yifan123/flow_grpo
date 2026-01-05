# 8 GPU
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_gardo.py --config config/grpo_gardo.py:hps_flux
