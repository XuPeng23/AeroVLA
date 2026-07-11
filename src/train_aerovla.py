import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
import torch
import transformers
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer, 
    AutoProcessor, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from aerovla_dataset import AeroVLADataset

# ------------------------------ Config --------------------------------
MODEL_PATH = "./openvla-7b"
DATA_ROOT = "./dataset_raw"
SPLIT_JSON = "./data/aerovla_train_dataset.json"
OUTPUT_DIR = "./checkpoints/aero_vla_r64_b99_bs64_lr2e-4_5epoch"

# ------------------------- Hyperparameters ---------------------------
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-4
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
NUM_EPOCHS = 5

def main():
    
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    device_map = {"": local_rank}
    
    print("Using Native BF16 (LoRA)...")
    torch_dtype = torch.bfloat16

    print("Loading Tokenizer & Processor...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if tokenizer.eos_token is None:
        print("Warning: EOS token is None, appending </s> manually.")
        tokenizer.eos_token = "</s>"

    if tokenizer.pad_token is None:
        NUM_BINS = 99
        print("Info: Pad token is None, setting to EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Model...")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    model.resize_token_embeddings(len(tokenizer))
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["projector"]
    )
        
    model = get_peft_model(model, lora_config)
    
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if "projector" in name:
                print(f"TRAINABLE: {name}")
    
    print(f"\nSummary: Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.4f}%")

    dataset = AeroVLADataset(DATA_ROOT, SPLIT_JSON, tokenizer, processor)
    
    def data_collator(features):
        
        pixel_values = torch.stack([f["image"] for f in features]).to(torch.bfloat16) 
        
        texts = [f["text"] for f in features]
        prompts = [f["prompt_only"] for f in features]
        
        batch = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=2048
        )
        
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        labels = input_ids.clone()
        
        for i, prompt in enumerate(prompts):
            prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids[0]
            prompt_len = len(prompt_tokens)
            
            labels[i, :prompt_len] = -100
            
        labels[attention_mask == 0] = -100
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.03, 
        warmup_ratio=0.03, 
        max_grad_norm=1,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        logging_steps=50,
        save_steps=2000,
        save_total_limit=5,
        num_train_epochs=NUM_EPOCHS,
        fp16=False,
        bf16=True, 
        report_to="tensorboard",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("Starting Training...")
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training Finished! Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
