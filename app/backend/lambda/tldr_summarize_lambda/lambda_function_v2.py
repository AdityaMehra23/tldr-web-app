import json
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

def lambda_handler(event, context):
    print(event)

    decoded_event = json.loads(event['body'])
    
    input_text = decoded_event['test_string']

    def clean_text(text):

        # replace ° degree with
        text = re.sub(r"°", " degree", text)

        # remove et al.
        text = re.sub(r'\bet al\. ,\s*', '', text)

        text = re.sub(r"FIG.","FIG", text)

        text = re.sub(r'Figure\s+\d+[A-Za-z]*', '', text)

        # replace semi-colon with full-stop
        text = re.sub(r";", ".", text)
        # Remove special characters and symbols
        text = re.sub(r"[^a-zA-Z0-9\s.:,-]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        # Split the paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Remove sentences containing "FIG."
        filtered_sentences = [sentence for sentence in sentences if "FIG" not in sentence and "FIGURE" not in sentence]

        # Join the remaining sentences to form the updated paragraph
        text = ' '.join(filtered_sentences)

        text = text.strip()

        return text
    
    text_to_filter = "".join(["".join(sentence) for sentence in input_text])
    cleaned_text = clean_text(text_to_filter)

    def clean_predicted_text(text):

        # replace semi-colon with full-stop
        text = re.sub(r";", ".", text)

        #replace new line character
        text = re.sub(r"<n>", "\n", text)

        #replace start of paragraph
        text = re.sub(r"<s>", "", text)

        # Remove special characters and symbols
        text = re.sub(r"[^a-zA-Z0-9\s.:,-]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        text = re.sub(r"FIG.","FIG", text)

        text = re.sub(r'Figure\s+\d+[A-Za-z]*', '', text)

        text = re.sub(r"\s+", " ", text)

        # Split the paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Remove sentences containing "FIG."
        filtered_sentences = [sentence for sentence in sentences if "FIG" not in sentence and "FIGURE" not in sentence]

        # Join the remaining sentences to form the updated paragraph
        text = ' '.join(filtered_sentences)

        text = re.sub(r'\bet al\. ,\s*', '', text)

        last_full_stop_index = text.rfind('.')

        if last_full_stop_index != -1:
            # Extract the substring up to the last full stop (including the full stop)
            text = text[:last_full_stop_index + 1]

        return text

    tokenizer = AutoTokenizer.from_pretrained("./model/")

    model = AutoModelForSeq2SeqLM.from_pretrained("./model/")

    summarizer = pipeline("summarization",model=model,tokenizer=tokenizer,num_beams=3,no_repeat_ngram_size=5,top_k=15)
    prediction = clean_predicted_text(summarizer(cleaned_text,min_length=150,max_length=450,do_sample=True)[0]["summary_text"])

    

    return {
        'statusCode': 200,
        'body': json.dumps(prediction)
    }
