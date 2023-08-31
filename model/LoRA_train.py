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

device = "cuda" if torch.cuda.is_available() else "cpu"

#Loading the Model
def get_tokenzier_model(model_ckpt):

    """
    returns the tokenizer and the model for a specific model checkpoint

    model_ckpt: model checkpoint name

    """
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if model_ckpt == "google/long-t5-tglobal-base":
      model = LongT5ForConditionalGeneration.from_pretrained(model_ckpt, torch_dtype=torch.bfloat16)
      print("success")
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

    return model, tokenizer

model, tokenizer = get_tokenzier_model(dict_model["model_ckpt_longt5_globalbase"])



#Getting Data
#Code for getting the F split of Big Patent Dataset since it has the minimum average max_length
from datasets import load_dataset
data = load_dataset("big_patent", "f")

# #Working on only 10k records          # Traning on Sonic for Full data
# # Subset 'n' records from the training set
# data["train"] = data["train"].shuffle(seed=42).select(range(10000))

# # Subset 'n' records from the test set
# data["test"] = data["test"].shuffle(seed=42).select(range(2000))

#Cleaner Functions
#Data Cleaning
def remove_sentence_with_fig(paragraph):


    # first we remove . after FIG
    paragraph = re.sub(r"FIG.","FIG",paragraph)

    # Split the paragraph into sentences
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)

    # Remove sentences containing "FIG."
    filtered_sentences = [sentence for sentence in sentences if "FIG" not in sentence]

    # Join the remaining sentences to form the updated paragraph
    updated_paragraph = ' '.join(filtered_sentences)

    return updated_paragraph

def clean_text(text):

    # replace ° degree with
    text = re.sub(r"°", " degree", text)

    # replace US Patent
    text = re.sub(r" U.S. Pat. No. [0-9,]+", "Patent", text)

    # replace semi-colon with full-stop
    text = re.sub(r";", ".", text)

    # Remove special characters and symbols
    text = re.sub(r"[^a-zA-Z0-9\s.:]", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    text = remove_sentence_with_fig(text)

    return text

#Cleaning
# Clean the 'description' column in the training set
cleaned_train_descriptions = [clean_text(desc) for desc in data["train"]["description"]]
data["train"] = Dataset.from_dict({"description": cleaned_train_descriptions, "abstract": data["train"]["abstract"]})


cleaned_test_descriptions = [clean_text(desc) for desc in data["validation"]["description"]]
data["validation"] = Dataset.from_dict({"description": cleaned_test_descriptions, "abstract": data["validation"]["abstract"]})


cleaned_test_descriptions = [clean_text(desc) for desc in data["test"]["description"]]
data["test"] = Dataset.from_dict({"description": cleaned_test_descriptions, "abstract": data["test"]["abstract"]})


# Tokenizer
#Tokenizing the input for Patents
prefix = "summarize: "



def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["description"]]
    # model_inputs = tokenizer(inputs, max_length=16384, truncation=True)
    model_inputs = tokenizer(inputs, max_length=8000, truncation=True)
    labels = tokenizer(text_target=examples["abstract"], max_length=200, truncation=True) #'description', 'abstract'

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    
tokenized_ds = data.map(preprocess_function, batched=True)
  
tokenized_ds = tokenized_ds.remove_columns(column_names=["description","abstract"])  #Removing Original cols


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


#Setting Up LORA
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

#Traning Adapter


output_dir  = f'lora_training_{str(int(time.time()))}'

peft_training_args = Seq2SeqTrainingArguments(
    output_dir  = output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=200,
    evaluation_strategy="epoch",
    max_steps=1,
    weight_decay=0.01,
    predict_with_generate=True,
    optim="adamw_torch_fused",
    save_strategy="epoch",
)

peft_trainer=Seq2SeqTrainer(

    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator
)



#Traning
peft_trainer.train()

peft_model_path = f"lora_training_checkpoints{str(int(time.time()))}"

peft_trainer.save_model(peft_model_path+"/Model")
peft_trainer.model.save_pretrained(peft_model_path)

tokenizer.save_pretrained(peft_model_path)





