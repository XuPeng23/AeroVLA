import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
from scipy.spatial.transform import Rotation as R
import re
from peft import PeftModel
import tkinter as tk
from PIL import ImageTk
from src.model_wrapper.base_model import BaseModelWrapper 


class AerialVLAWrapper(BaseModelWrapper):

    def __init__(self, model_args, data_args):

        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adapter_path = model_args.model_path 
        base_model_path = "./openvla-7b"
        print(f"[AerialVLA] Loading BASE from: {base_model_path}")
        print(f"[AerialVLA] Loading ADAPTER/TOKENIZER from: {adapter_path}")
        self.NUM_BINS = 99
        self.norm_stats = {
            'forward': {'min': 0.0, 'max': 5.0},
            'down':    {'min': -5.0, 'max': 5.0},
            'yaw':     {'min': -1.1, 'max': 1.1}
        }

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        # self.model = self.model.merge_and_unload()
        self.model.to(self.device)
        self.model.eval()

    def get_semantic_direction(self, curr_state, target_pos):
        
        pos = np.array(curr_state['position'])
        q_raw = curr_state['orientation']
        if isinstance(q_raw, dict):
            quat = [q_raw.get('x',0), q_raw.get('y',0), q_raw.get('z',0), q_raw.get('w',1)]
        else:
            quat = q_raw

        vec_world = np.array(target_pos) - pos

        dist_xy = np.linalg.norm(vec_world[:2])
        
        r = R.from_quat(quat)
        vec_body = r.inv().apply(vec_world)
        
        x, y = vec_body[0], vec_body[1]
        angle_deg = np.degrees(np.arctan2(y, x))

        if dist_xy < 0.01:
            return ""
        else:
            if -15 <= angle_deg <= 15: return "straight ahead "
            if 15 < angle_deg <= 60: return "forward-right "
            if 60 < angle_deg <= 120: return "to your right "
            if 120 < angle_deg <= 180: return "to your right rear "
            if -60 <= angle_deg < -15: return "forward-left "
            if -120 <= angle_deg < -60: return "to your left "
            if -180 <= angle_deg < -120: return "to your left rear "

        return ""

    def prepare_inputs(self, episodes, target_positions, instructions=None):

        prompts = []
        pixel_values_list = []
        
        for i, ep in enumerate(episodes):
            curr_state = ep[-1]['sensors']['state'] 
            rgb_list = ep[-1]['rgb'] 
            frame_front_rgb = cv2.cvtColor(rgb_list[0], cv2.COLOR_BGR2RGB)
            frame_down_rgb = cv2.cvtColor(rgb_list[4], cv2.COLOR_BGR2RGB)
            img_front = Image.fromarray(frame_front_rgb)
            img_down = Image.fromarray(frame_down_rgb)

            target_pos = target_positions[i]
            dir_text = self.get_semantic_direction(curr_state, target_pos)
            obj_desc = instructions[i].split("degrees from you.")[1].split(" Please control")[0].strip()

            # ================== UI START ====================================================
            # d_front = cv2.copyMakeBorder(cv2.resize(cv2.cvtColor(np.array(img_front), cv2.COLOR_RGB2BGR), (256, 128)), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            # d_down = cv2.copyMakeBorder(cv2.resize(cv2.cvtColor(np.array(img_down), cv2.COLOR_RGB2BGR), (256, 128)), 0, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            # view_panel = np.vstack((d_front, d_down))

            # txt_panel = cv2.copyMakeBorder(np.zeros((120, 420, 3), dtype=np.uint8), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            # words = f"Prompt: Fly {dir_text.strip() if dir_text.strip() else 'nearby'} and find the target. {obj_desc}".split()
            # x, y = 15, 28
            # for w in words:
            #     c = (0, 0, 255) if w in (dir_text.strip() if dir_text.strip() else "nearby").split() else (255, 255, 255)
            #     tw = cv2.getTextSize(w, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            #     if x + tw > 405: x, y = 15, y + 22
            #     cv2.putText(txt_panel, w, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
            #     x += tw + 6

            # act_panel = cv2.copyMakeBorder(np.zeros((35, 420, 3), dtype=np.uint8), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            # cv2.putText(act_panel, "Action: Inferring...", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

            # if not hasattr(self, 'tk_roots'): self.tk_roots, self.tk_lbls = {}, {}
            # if i not in self.tk_roots:
            #     r = tk.Tk()
            #     r.withdraw()
            #     self.tk_roots[i], self.tk_lbls[i] = r, []
            #     for geo in ["+1745+1085", "+1170+655", "+1170+785"]:
            #         top = tk.Toplevel(r)
            #         top.overrideredirect(True)
            #         top.attributes('-topmost', True)
            #         top.geometry(geo)
            #         lbl = tk.Label(top, borderwidth=0, highlightthickness=0)
            #         lbl.pack()
            #         self.tk_lbls[i].append(lbl)

            # for idx, panel in enumerate([view_panel, txt_panel, act_panel]):
            #     img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)))
            #     self.tk_lbls[i][idx].configure(image=img)
            #     self.tk_lbls[i][idx].image = img
            # self.tk_roots[i].update()
            # ================== UI END ======================================================
            

            target_size = (224, 224)
            img_front = img_front.resize(target_size, resample=Image.BICUBIC)
            img_down = img_down.resize(target_size, resample=Image.BICUBIC)
            w, h = target_size
            mosaic = Image.new('RGB', (w, h * 2), (0, 0, 0))
            mosaic.paste(img_front, (0, 0))
            mosaic.paste(img_down, (0, h))
            
            pv = self.image_processor(images=mosaic, return_tensors='pt')['pixel_values'].squeeze(0)
            pixel_values_list.append(pv)

            prompt_text = (
                f"<image>\n"
                f"Fly {dir_text}and find the target. {obj_desc}\n"
                f"Action: "
            )
            prompts.append(prompt_text)

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        pixel_values = torch.stack(pixel_values_list).to(self.device)

        if hasattr(self.model, "dtype"): 
            pixel_values = pixel_values.to(self.model.dtype)
            
        inputs['pixel_values'] = pixel_values
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs, None


    def run(self, inputs, episodes, rot_to_targets):

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False,
                eos_token_id=[self.tokenizer.eos_token_id]
            )
            
        batch_actions = []
        batch_should_stop = []

        for i, output_ids in enumerate(generated_ids):
            
            text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
            pred_fwd, pred_down, pred_yaw = self._parse_action_from_text(text)
            raw_action_str = text.split("Action:")[-1].replace("</s>", "").replace("<pad>", "").strip()

            #  ================== UI START ====================================================
            # if hasattr(self, 'tk_lbls') and i in self.tk_lbls:
            #     act_panel = cv2.copyMakeBorder(np.zeros((35, 420, 3), dtype=np.uint8), 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(220, 11, 120))
            #     cv2.putText(act_panel, f"Action: {raw_action_str}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            #     img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(act_panel, cv2.COLOR_BGR2RGB)))
            #     self.tk_lbls[i][2].configure(image=img)
            #     self.tk_lbls[i][2].image = img
            #     self.tk_roots[i].update()
            #  ================== UI END ======================================================
            
            has_land_str = "LAND" in text or "<LAND>" in text
            has_land_num = (pred_fwd < 0.01) and (abs(pred_down) < 0.01) and (abs(pred_yaw) < 0.01)
            has_land = has_land_str or has_land_num
            batch_should_stop.append(has_land)

            batch_actions.append({
                'fwd': pred_fwd,
                'down': pred_down,
                'yaw': pred_yaw
            })
            
        return batch_actions, batch_should_stop
    

    def _parse_action_from_text(self, text):
        
        output_part = text.split("Action:")[-1]
        matches = re.findall(r"\d+", output_part)
        fwd, down, yaw = 0.0, 0.0, 0.0
        
        if len(matches) >= 3:
            try:
                bin_fwd, bin_down, bin_yaw = map(int, matches[-3:])
                def dequantize(bin_val, axis):
                    vmin, vmax = self.norm_stats[axis]['min'], self.norm_stats[axis]['max']
                    return (max(0, min(self.NUM_BINS - 1, bin_val)) / (self.NUM_BINS - 1)) * (vmax - vmin) + vmin
                fwd, down, yaw = dequantize(bin_fwd, 'forward'), dequantize(bin_down, 'down'), dequantize(bin_yaw, 'yaw')
            except:
                pass
        return fwd, down, yaw
