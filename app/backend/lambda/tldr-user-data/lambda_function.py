import json

from modules.user_data import *

def lambda_handler(event, context):
    
    req_body = json.loads(event['body'])['parameters']


    res = {}
    username = req_body['username']
    
    if(req_body['function'] == 'save_data'):
        user_input = req_body['user_input']
        summary = req_body['summary']
        summary_type = req_body['summary_type']
        res = store_user_data(username, user_input, summary,summary_type)
    elif(req_body['function'] == 'all_data'):
        res = fetch_all_user_records(username)
    elif(req_body['function'] == 'single_record'):
        timestamp = req_body['timestamp']
        res = fetch_single_user_record(username,timestamp)
        
            
    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }