import json
from src.users import *

def lambda_handler(event, context):
    
    req_body = json.loads(event['body'])['parameters']
    
    res = {}
    
    if(event['httpMethod'] == 'POST'):
        if(req_body['function'] == 'register'):
            username = req_body['username']
            password = req_body['password']
            email = req_body['email']
            print("register: " + username + '__' + password + '__' + email)
            res = register_user(username, password, email)
        elif(req_body['function'] == 'verify-user'):
            username = req_body['username']
            code = req_body['code']
            print("verify-user: " + username + '__' + code)
            res = verify_user(username, code)
        elif(req_body['function'] == 'sign-in'):
            username = req_body['username']
            password = req_body['password']
            print("sign-in: " + username + '__' + password)
            res = sign_in(username, password)
        elif(req_body['function'] == 'logout'):
            refresh_token = req_body['refresh_token']
            print("logout: " + refresh_token)
            res = logout(refresh_token)
        elif(req_body['function'] == 'forgot-password'):
            username = req_body['username']
            print("forgot-password: " + username)
            res = forgot_password(username)
        elif(req_body['function'] == 'confirm-forgot-password'):
            username = req_body['username']
            confirmation_code = req_body['confirmation-code']
            password = req_body['password']
            print("confirm-forgot-password: " + username + '_' + password + '_' + confirmation_code)
            res = confirm_forgot_password(username, password, confirmation_code)
            
    return {
        'statusCode': 200,
        'body': json.dumps(res)
    }