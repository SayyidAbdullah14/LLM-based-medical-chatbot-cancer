from peft import PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    GenerationConfig
    )
import pandas as pd
import torch
from datasets import load_dataset, concatenate_datasets

#Load model
base_model= AutoModelForCausalLM.from_pretrained(
    "BioMistral/BioMistral-7B",
    load_in_8bit=True,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "sayyid14/BioMistralCancer-withLoRA")
generation_config = GenerationConfig(
    do_sample=False,
    #max_lenght=20,
    #temperature=0.1,
    #top_p=0.9,
    #top_k=10,
    num_beams=4,
    repetition_penalty=1.15,
)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def generate_response(question):
    PROMPT = f"""Below is a question about cancer.  Please answer this question correctly. if you don't knwo the answer just say that you don't know and don't share false information.
### Question:
{question}
### Answer:"""

    inputs = tokenizer(
        PROMPT,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"]#.cuda()

    print("Generating...")
    generation_output = model.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    return_dict_in_generate=True,
    output_scores=True,
    max_new_tokens=10,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
    for s in generation_output.sequences:
        result = tokenizer.decode(s).split("### Answer:")[1]

    return result

import pandas as pd

#Importing the dataset
dataset_name = "sayyid14/PubMedMCQACancer"
dataset = load_dataset(dataset_name, split="train[0:500]")

df=pd.DataFrame(dataset)
df=df.dropna()

df['predicted']=df["question"].str.len()
for i in range(499):
  question= str(df['question'].iloc[i-1]+ ' yes or no?')
  print("Question : " + question)
  hasil=generate_response(question)
  print("Answer : " + df['final_decision'].iloc[i-1])
  print("Predicted Answer : " + hasil)
  df['predicted'].iloc[i-1]=hasil

df.to_csv("hasiltestBioMistralCancertoPubMedQA2.csv")