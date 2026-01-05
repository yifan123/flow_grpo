import os
import torch
import numpy as np
from PIL import Image
import huggingface_hub
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class HPSv2RewardInferencer:
    """
    HPSv2 Reward Inferencer for scoring image-text alignment.
    
    This scorer uses CLIP (ViT-H-14) to compute cosine similarity between
    image and text features as a reward signal.
    """
    
    def __init__(self, device='cuda', dtype=torch.float32, hps_version='v2.1'):
        """
        Initialize the HPSv2 reward inferencer.
        
        Args:
            device (str): Device to run inference on ('cuda' or 'cpu')
            dtype: Data type for model weights (torch.float32, torch.float16, etc.)
            hps_version (str): Version of HPS model ('v2.0' or 'v2.1')
        """
        # Import here to avoid dependency issues if not using HPSv2
        import sys
        hpsv2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'HPSv2')
        if hpsv2_path not in sys.path:
            sys.path.insert(0, hpsv2_path)
        
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        
        self.device = device
        self.dtype = dtype
        self.hps_version = hps_version
        
        # Version mapping
        hps_version_map = {
            "v2.0": "HPS_v2_compressed.pt",
            "v2.1": "HPS_v2.1_compressed.pt",
        }
        
        # Create model and transforms
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        
        # Download checkpoint from HuggingFace
        checkpoint_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2", 
            hps_version_map[hps_version]
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Get tokenizer
        tokenizer = get_tokenizer('ViT-H-14')
        
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        self.model = model
        self.preprocess_val = preprocess_val
        self.tokenizer = tokenizer
    
    @torch.inference_mode()
    def reward(self, images, prompts):
        """
        Calculate HPSv2 reward scores for image-prompt pairs.
        
        Args:
            images: List of PIL Images or file paths
            prompts: List of text prompts
            
        Returns:
            np.ndarray: Array of reward scores (cosine similarity)
        """
        scores = []
        
        # Process each image-prompt pair
        for image, prompt in zip(images, prompts):
            # Handle different input types
            if isinstance(image, str):
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif not isinstance(image, Image.Image):
                raise TypeError(f'Unsupported image type: {type(image)}')
            
            # Preprocess image
            image_tensor = self.preprocess_val(image).unsqueeze(0).to(
                device=self.device, non_blocking=True
            )
            
            # Tokenize text
            text_tensor = self.tokenizer([prompt]).to(
                device=self.device, non_blocking=True
            )
            
            # Calculate HPS score
            with torch.cuda.amp.autocast():
                outputs = self.model(image_tensor, text_tensor)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
            
            scores.append(hps_score[0])
        
        return np.array(scores)


if __name__ == "__main__":
    # Test the scorer
    import sys
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize scorer
    print("Initializing HPSv2 scorer...")
    scorer = HPSv2RewardInferencer(device=device, hps_version='v2.1')
    
    # Test with example images (if available)
    test_prompts = [
        "A cat with two horns on its head",
    ]
    
    print("Scorer initialized successfully!")
    print(f"Model device: {scorer.device}")
    print(f"HPS version: {scorer.hps_version}")