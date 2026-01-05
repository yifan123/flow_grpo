import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.num_epochs = 100
    config.use_lora = True
    config.train.algorithm = 'gardo'
    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    config.reset_freq = 15
    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    config.kl_thres=3e-4 #Set up the threshold according to specific tasks
    return config

def general_ocr_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5
    # config.train.lora_path=''
    # # 8*A800
    # config.resolution = 512
    # config.sample.train_batch_size = 12
    # config.sample.num_image_per_prompt = 24
    # config.sample.num_batches_per_epoch = 12
    # config.sample.test_batch_size = 16 # 11 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    # 1 A800 This is just to ensure it runs quickly on a single GPU, though the performance may degrade.
    # If using 8 GPUs, please comment out this section and use the 8-GPU configuration above instead.
    gpu_number = 8
    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 16 # 11 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.89
    # kl loss
    config.train.beta = 0.04#0.004
    # kl reward
    # KL reward and KL loss are two ways to incorporate KL divergence. KL reward adds KL to the reward, while KL loss, introduced by GRPO, directly adds KL loss to the policy loss. We support both methods, but KL loss is recommended as the preferred option.
    config.sample.kl_reward = 0
    # We also support using SFT data in RL training for supervised learning to prevent quality drop, but this option was unused
    config.train.sft=0.0
    config.train.sft_batch_size=3
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=True
    config.train.ema=True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.num_epochs = 100000
    config.run_name = 'flow_grpo_gardo_ocr'
    config.train.clip_range_low = 0.00001
    config.train.clip_range_high = 0.00001
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.main_reward = 'ocr'
    config.save_dir = './flow_grpo_gardo_ocr'
    config.reward_fn = {
        # "geneval": 1.0,
        "ocr": 0.8,
        "aesthetic":0.1,
        "imagereward": 0.1,
        # "imagereward":1.0
        # "hpsv3_score":1.0,
        # "unifiedreward": 0.7,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def geneval_sd3():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    gpu_number = 8
    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10 # 40
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale=4.5
    # config.train.lora_path=''
    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.89
    config.train.beta = 0.07#0.004
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.seed=42
    config.run_name = 'flow_grpo_gardo_geneval'
    config.save_dir = './flow_grpo_gardo_geneval'
    config.reward_fn = {
        "geneval": .8,
        "aesthetic":0.05,
        "imagereward": 0.15,
        # "unifiedreward": 0.7,
    }
    config.main_reward = 'geneval'
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def hps_flux():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    gpu_number=8
    # sd3.5 medium
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6 # 40
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale=3.5
    # config.train.lora_path=''
    # 8 cards to start LLaVA Server
    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.7
    config.train.beta = 0.01#0.004
    config.sample.kl_reward = 0
    config.sample.global_std=True
    config.train.ema=True
    config.mixed_precision = "bf16"
    config.num_epochs = 100000
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.sample.same_latent=True
    config.sample.noise_level=0.9
    config.run_name = 'flow_grpo_gardo-flux'
    config.save_dir = './flow_grpo_gardo_flux'
    config.reward_fn = {
        "hpsv2_score":0.8,
        "aesthetic":0.05,
        "imagereward": 0.15,
        # "geneval": 1.0,
        # "ocr": 1.0,
        # "unifiedreward": 0.7,
    }
    config.main_reward = 'hpsv2_score'
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config



def get_config(name):
    return globals()[name]()
