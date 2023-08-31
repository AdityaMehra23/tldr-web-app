"""
This training script can be run both on a single gpu 

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

"""


import pandas as pd
import numpy as np
from datasets import load_dataset, load_metric
from transformers import pipeline, set_seed
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LongT5ForConditionalGeneration, DataCollatorForSeq2Seq


from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tqdm import tqdm
import re
import torch 
from datasets import Dataset

import evaluate


#Evaluation metric
rouge = evaluate.load("rouge")


#Dictionary for Model checkpoints
dict_model ={}
dict_model["model_ckpt_longt5_globalbase"] = "google/long-t5-tglobal-base"

train_file_path = "train_research.csv"
model_path =""

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_tokenzier_model(model_ckpt):

    """
    returns the tokenizer and the model for a specific model checkpoint

    model_ckpt: model checkpoint name

    """
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    if model_ckpt == "google/long-t5-tglobal-base":
      model = LongT5ForConditionalGeneration.from_pretrained(model_ckpt).to(device)
      print("success")
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

    return model, tokenizer


# Helper functions to clean the data


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

def getDataNRows(filename,cleanCol, n=100):
    """
    Gets the data in dataset format for n rows of  csv file

    filename: full path to the csv file

    n: Number of Rows needed

    """

    d = pd.read_csv(filename)
    d = d.head(n)
    d[cleanCol] = d[cleanCol].map(clean_text)
    dataset = Dataset.from_pandas(d)

    return dataset


#Getting the data ready
data = getDataNRows(train_file_path,"sections",1000)

#splitting the data
ds = data.train_test_split(test_size=0.2)


#Loading the model
model, tokenizer = get_tokenzier_model(dict_model["model_ckpt_longt5_globalbase"])

#Tokenizing the input
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["sections"]]
    # model_inputs = tokenizer(inputs, max_length=16384, truncation=True)
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["summary"], max_length=200, truncation=False) #'sections', 'abstract', 'summary'

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = ds.map(preprocess_function, batched=True)

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


training_args = Seq2SeqTrainingArguments(
    output_dir="my__paperabstract_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(model_path)

