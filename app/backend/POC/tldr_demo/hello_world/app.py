import json
import torch
import numpy as np
from transformers import pipeline

llm = pipeline('summarization',model="/opt/ml/model",tokenizer="/opt/ml/model")

def lambda_handler(event, context):
    raw_string = r'{}'.format(event['body'])
    body = json.loads(raw_string)
    originaltext = body['text']
    model_input = "summarize: "+originaltext
    result = llm(model_input)
       
    return {
        "statusCode": 200,
        "body": json.dumps(result),
    }
