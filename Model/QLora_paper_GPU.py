from datasets import load_dataset , Dataset
from transformers import AutoModelForSeq2SeqLM, LongT5ForConditionalGeneration, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import re

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


#Evaluation metric
rouge = evaluate.load("rouge")


#Dictionary for Model checkpoints
dict_model ={}
dict_model["model_ckpt_longt5_globalbase"] = "google/long-t5-tglobal-base"

# train_file_path = "train_research.csv"
# model_path =""


from datasets import load_dataset
data = load_dataset("pszemraj/scientific_lay_summarisation-elife-norm")

# #Working on only 10k records          # Traning on Sonic for Full data
# # Subset 'n' records from the training set
# data["train"] = data["train"].shuffle(seed=42).select(range(10000))

# # Subset 'n' records from the test set
# data["test"] = data["test"].shuffle(seed=42).select(range(2000))

#Cleaner Functions
#Data Cleaning

import re

def clean_text(text: str) -> str:
    # Remove spaces before punctuation marks (except for parentheses)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Add a space after punctuation marks (except for parentheses) if missing
    text = re.sub(r"([.,;:!?])(?=[^\s])", r"\1 ", text)

    # Handle spaces around parentheses
    text = re.sub(r"\s?\(\s?", r" (", text)
    text = re.sub(r"\s?\)\s?", r")", text)

    # Add a space after a closing parenthesis if:
    # followed by a word or opening parenthesis
    text = re.sub(r"\)(?=[^\s.,;:!?])", r") ", text)

    # Handle spaces around quotation marks
    text = re.sub(r'\s?"', r'"', text)
    text = re.sub(r'"\s?', r'" ', text)

    # Handle spaces around single quotes
    text = re.sub(r"\s?'", r"'", text)
    text = re.sub(r"'\s?", r"' ", text)

    # Handle comma in numbers
    text = re.sub(r"(\d),\s+(\d)", r"\1,\2", text)

    return text


#Cleaning
# Clean the 'description' column in the training set
cleaned_train_descriptions = [clean_text(desc) for desc in data["train"]["article"]]
data["train"] = Dataset.from_dict({"article": cleaned_train_descriptions, "summary": data["train"]["summary"]})


cleaned_test_descriptions = [clean_text(desc) for desc in data["validation"]["article"]]
data["validation"] = Dataset.from_dict({"article": cleaned_test_descriptions, "summary": data["validation"]["summary"]})


cleaned_test_descriptions = [clean_text(desc) for desc in data["test"]["article"]]
data["test"] = Dataset.from_dict({"article": cleaned_test_descriptions, "summary": data["test"]["summary"]})



from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

#Loading the Model
def get_tokenzier_model(model_ckpt):

    """
    returns the tokenizer and the model for a specific model checkpoint

    model_ckpt: model checkpoint name

    """
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if model_ckpt == "google/long-t5-tglobal-base":
      model = LongT5ForConditionalGeneration.from_pretrained(model_ckpt, quantization_config=bnb_config)
      print("success")
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

    return model, tokenizer

model, tokenizer = get_tokenzier_model(dict_model["model_ckpt_longt5_globalbase"])

model.config.use_cache = False

#Tokenizing the input for Papers
prefix = "summarize: "



def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["article"]]
    # model_inputs = tokenizer(inputs, max_length=16384, truncation=True)
    model_inputs = tokenizer(inputs, max_length=8000, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=300, truncation=True) #'article', 'summary'

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    
tokenized_ds = data.map(preprocess_function, batched=True)
  
tokenized_ds = tokenized_ds.remove_columns(column_names=["article","summary"])  #Removing Original cols


#Data Collation
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=dict_model["model_ckpt_longt5_globalbase"])


def compute_metrics(eval_pred):
    """
    To compute metrics while training

    """
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Setting Up LORA
from peft import LoraConfig, get_peft_model, TaskType

lora_config=LoraConfig(
r=32, # Rank
lora_alpha=32,
target_modules=["q","v"],
lora_dropout=0.05,
bias="none",
task_type=TaskType.SEQ_2_SEQ_LM
)

peft_model = get_peft_model(model,lora_config)

output_dir  = f'lora_training_{str(int(time.time()))}'

peft_training_args = Seq2SeqTrainingArguments(
    output_dir  = output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    evaluation_strategy="steps",
    max_steps=100,
    save_steps=50, #Added Save Steps
    weight_decay=0.01,
    predict_with_generate=True,
    fp16=True,
    save_strategy="epoch",
)

peft_trainer=Seq2SeqTrainer(

    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)


#Traning
peft_trainer.train()

#file name 
import os
_file  = os.path.basename(__file__)
_file = _file[:-3]

peft_model_path = f"{_file}/lora_training_checkpoints{str(int(time.time()))}"

peft_trainer.save_model(peft_model_path+"/Model")
peft_trainer.model.save_pretrained(peft_model_path)
