import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

ACTION_STATS = {
    "forward": {"min": 0.0, "max": 5.0}, 
    "down":    {"min": -5.0, "max": 5.0}, 
    "yaw":     {"min": -1.1, "max": 1.1}  
}
NUM_BINS = 99

class AeroVLADataset(Dataset):
    def __init__(self, data_root, clean_json_path, tokenizer, processor):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.processor = processor
        
        with open(clean_json_path, 'r') as f:
            self.samples = json.load(f)
        print(f"[Dataset] Loaded {len(self.samples)} samples successfully.")

    def __len__(self):
        return len(self.samples)

    def _get_images(self, rel_dir, filename):
        images = []
        target_size = (224, 224)
        traj_full_dir = os.path.join(self.data_root, rel_dir)
        
        cam_names = {
            'front': ['frontcamera'],
            'down':  ['downcamera']
        }
        
        for cam_type in ['front', 'down']:
            img = None
            for folder_name in cam_names[cam_type]:
                img_path = os.path.join(traj_full_dir, folder_name, filename)
                if os.path.exists(img_path):
                    try:
                        i = Image.open(img_path).convert('RGB')
                        i = i.resize(target_size, resample=Image.BICUBIC)
                        img = i
                        break
                    except Exception as e:
                        print(f"ERROR LOADING IMAGE: {img_path} - {e}")
                        raise e
                else:
                    raise FileNotFoundError(f"CRITICAL: Image not found at {img_path}\nCheck DATA_ROOT config!")

            images.append(img)
            
        w, h = target_size
        mosaic = Image.new('RGB', (w, h * 2), (0, 0, 0))
        mosaic.paste(images[0], (0, 0)) 
        mosaic.paste(images[1], (0, h)) 
        return mosaic

    def _quantize_action(self, val, axis):
        vmin = ACTION_STATS[axis]['min']
        vmax = ACTION_STATS[axis]['max']
        val = np.clip(val, vmin, vmax)
        norm = (val - vmin) / (vmax - vmin)
        return int(norm * (NUM_BINS - 1))

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = self._get_images(sample['traj_rel_dir'], sample['img_name'])
        image_tensor = self.processor.image_processor(images=image, return_tensors='pt')['pixel_values'].squeeze(0)
        
        bin_fwd = self._quantize_action(sample['label']['fwd'], 'forward')
        bin_down = self._quantize_action(sample['label']['down'], 'down')
        bin_yaw = self._quantize_action(sample['label']['yaw'], 'yaw')
        
        action_str = f"{bin_fwd:02d} {bin_down:02d} {bin_yaw:02d}"
        if sample['is_last_step'] or sample['is_penultimate']:
            action_str += " LAND"
        
        if self.tokenizer.eos_token:
            action_str += self.tokenizer.eos_token
        else:
            print("Warning: EOS token is None, appending </s> manually.")
            action_str += "</s>"
            
        instr = sample['instruction'].strip()
        action = 'Action: '

        full_text = f"{instr}\n{action}{action_str}"
        prompt_only = f"{instr}\n{action}" 
        
        return {
            "image": image_tensor,
            "text": full_text,
            "prompt_only": prompt_only 
        }
        

