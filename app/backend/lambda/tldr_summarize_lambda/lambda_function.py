import json
from transformers import pipeline

def lambda_handler(event, context):
    print(event)

    decoded_event = json.loads(event['body'])
    
    input_text = decoded_event['test_string']

    summarizer = pipeline("summarization", model="./model/")
    summary= summarizer(input_text, max_length=70)

    print(summary)
    summarised = summary[0]["summary_text"]

    return {
        'statusCode': 200,
        'body': json.dumps(summarised)
    }
