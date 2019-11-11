import json
import datetime


def handler(event, context):
    body = {}
    if event['testpost']:
        try:
            body = json.loads(event['testpost']['body'])
        except:
            return {'statusCode': 400,
                    'body': 'Invalid input! Expecting a JSON.',
                    'headers': {'Content-Type': 'application/json'}}

    data = {
        'output': body,
        'timestamp': datetime.datetime.utcnow().isoformat()
    }
    return {'statusCode': 200,
            'body': json.dumps(data),
            'headers': {'Content-Type': 'application/json'}}
