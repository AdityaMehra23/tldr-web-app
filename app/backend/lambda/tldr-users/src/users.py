import boto3
from botocore.exceptions import ClientError

client = boto3.client('cognito-idp', region_name='eu-west-1')
CLIENT_ID = 'nlq4k1jr29nbr7q41dh1s5jj9'
USER_POOL_ID = 'eu-west-1_YwRWiXjhg'


def register_user(username, password, email):
    # secret_hash = get_secret_hash(username)
    
    username = username.strip()
    password = password.strip()
    email = email.strip()
    
    return_statement = {}
    
    try:
        return_statement["resp"] = client.sign_up(
                ClientId=CLIENT_ID,
                Username=username,
                Password=password,
                UserAttributes=[
                {
                    'Name': "email",
                    'Value': email
                }
                ])
    except ClientError as e:
        if e.response['Error']['Code'] == 'UsernameExistsException':
            return_statement["error"] = "Username already exists."
        else:
            return_statement["error"] = "Something went wrong."
    
    return return_statement
    
def verify_user(username, code):
    
    username = username.strip()
    code = code.strip()
    
    return_statement = {}
    
    try:
        return_statement["resp"] = client.confirm_sign_up(
                ClientId=CLIENT_ID,
                Username=username,
                ConfirmationCode=code)
    except ClientError as e:
        if e.response['Error']['Code'] == 'CodeMismatchException':
            return_statement["error"] = "Code Mismatch"
        else:
            return_statement["error"] = "Something went wrong."
    
    return return_statement
    
def sign_in(username, password):
    
    username = username.strip()
    password = password.strip()
    
    return_statement = {}
    try:
        return_statement["resp"] = client.initiate_auth(
                ClientId=CLIENT_ID,
                AuthFlow='USER_PASSWORD_AUTH',
                AuthParameters={
                         'USERNAME': username,
                         'PASSWORD': password
                      })
    except ClientError as e:
        if e.response['Error']['Code'] == 'NotAuthorizedException':
            return_statement["error"] = "Incorrect username or password."
        else:
            return_statement["error"] = "Something went wrong."
    
    return return_statement
    
def logout(refresh_token):
    
    resp = client.revoke_token(
            Token=refresh_token,
            ClientId=CLIENT_ID
        )
    return resp

    
def forgot_password(username):
    
    username = username.strip()
    return_statement = {}
    try:
        return_statement["resp"] = client.forgot_password(
                ClientId=CLIENT_ID,
                Username=username)
    except ClientError as e:
        if e.response['Error']['Code'] == 'UserNotFoundException':
            return_statement["error"] = "User not found."
        elif e.response['Error']['Code'] == 'CodeDeliveryFailureException':
            return_statement["error"] = "Code Delivery Failure."
        else:
            return_statement["error"] = "Something went wrong."
    
    return return_statement
    
def confirm_forgot_password(username, password, confirmation_code):
    
    username = username.strip()
    password = password.strip()
    confirmation_code = confirmation_code.strip()
    
    return_statement = {}
    try:
        return_statement["resp"] = client.confirm_forgot_password(
                ClientId=CLIENT_ID,
                Username=username,
                ConfirmationCode=confirmation_code,
                Password=password
                )
    except ClientError as e:
        if e.response['Error']['Code'] == 'InvalidPasswordException':
            return_statement["error"] = "Invalid Password."
        elif e.response['Error']['Code'] == 'CodeMismatchException':
            return_statement["error"] = "Code Mismatch."
        elif e.response['Error']['Code'] == 'UserNotFoundException':
            return_statement["error"] = "User not found."
        else:
            return_statement["error"] = "Something went wrong."
    
    return return_statement