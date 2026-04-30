from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch
from datasets import load_dataset
from trl import SFTTrainer
import wandb

torch.cuda.empty_cache()

dataset_name = "sayyid14/cancerdatasets"
df = load_dataset(dataset_name, split="train[0:36400]")

def format_text(example):
    text = f"instruction: {example['instruction']} question: {example['input']} answer: {example['output']}"
    # Remove "abstract" and "title" if present
    text = text.replace("abstract", "").replace("title", "")
    return {
        'formatted_text': text
    }
# Menggunakan map untuk menerapkan fungsi pada setiap contoh
formatted_dataset = df.map(format_text)

#df['formatted_text'] = df.apply(lambda row: f"instruction: {row['instruction']} question: {row['input']} answer: {row['output']}", axis=1)


import os
# Set secret token
os.environ['HF_TOKEN'] = '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'

# login wandb
wandb.init(project='tesiscolab', entity='radenmassayyidabdullah1408')

base_model = "BioMistral/BioMistral-7B"
new_model = "BioMistralCancer"

# Load base model
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense',
        'fc1',
        'fc2',
    ]
)
model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./BioMistralCancer2311LoRA5",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=1000,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    disable_tqdm=False,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    max_seq_length= 2048,
    dataset_text_field="formatted_text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)


trainer.train()

# Save trained model
wandb.save('mode2311.pth')
trainer.model.save_pretrained("BioMistralCancer2311LoRA5")
wandb.finish()