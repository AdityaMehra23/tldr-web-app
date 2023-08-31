import boto3
from botocore.exceptions import ClientError
import json
import time
import pandas as pd

client = boto3.client('dynamodb')

def store_user_data(username, user_input, summary,summary_type):
    
    timestamp = str(int(time.time()))
    
    resp = client.put_item(
        TableName = 'user_history',
        Item = {
            'username' : {'S' : username},
            'timestamp' : {'N' : timestamp},
            'user_input' : {'S' : user_input},
            'summary' : {'S' : summary},
            'summary_type': {'S' : summary_type}
        })
    
    return resp
    
def fetch_all_user_records(username):
    """
        return a list of all records for user, list of dictionaries
    """
    out = [{"summary_type":"","summary": "", "username": username, "user_input": "", "timestamp": ""}]
    try:
        response = client.query(
            TableName='user_history',
            KeyConditionExpression='username = :username',
            ExpressionAttributeValues={
                ':username': {'S': username}
            }
        )
        # If no history for user
        if response['Count']==0:
            raise Exception("No data found")
        df = pd.DataFrame(response['Items'])
        if 'summary_type' in df.columns:
            df['summary_type'] = df['summary_type'].str['S']
        df['summary'] = df['summary'].str['S']
        df['username'] = df['username'].str['S']
        df['user_input'] = df['user_input'].str['S']
        df['timestamp'] = df['timestamp'].str['N']
        def get_first_4_words(paragraph):
            words = paragraph.split()[:4]
            return " ".join(words)+"..."
        df['user_input'] = df['user_input'].apply(get_first_4_words)
        df['summary'] = df['summary'].apply(get_first_4_words)
        out = df.to_dict(orient='records')
    except ClientError as c:
        if c.response['Error']['Code'] == 'ResourceNotFoundException':
            print("The table was not found.")
    except Exception as e:  
            print("No data for the user")
            
    return out
    
def fetch_single_user_record(username, timestamp):
    """
        returns specific record for user and timestamp as a dictionary
    """
    out = {'summary_type':"",'username':username, 'timestamp':timestamp, 'user_input':"", "summary":"" }
    try:
        resp = client.get_item(
            TableName = 'user_history', 
            Key = {
                'username' : {'S' : username},
                'timestamp' : {'N' : timestamp}
            })
        
        if resp.get('Item','$%^')=='$%^':
            raise Exception("No data for the user and timestamp")
        if 'summary_type' in resp['Item'].keys():
            out['summary_type'] = resp['Item']['summary_type']['S']
        out['user_input'] = resp['Item']['user_input']['S']
        out['summary'] = resp['Item']['summary']['S']
    except ClientError as c:
        if c.response['Error']['Code'] == 'ResourceNotFoundException':
            print("The requested table was not found.")
    except Exception as e:  
            print("No data for the user and timestamp")
    return out
    
    