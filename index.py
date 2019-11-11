import json
import datetime


def handler(event, context):
    body = {}
    print(event)
    body = event['testpost']

    data = {
        'output': body,
        'timestamp': datetime.datetime.utcnow().isoformat()
    }
    return {'statusCode': 200,
            'body': json.dumps(data),
            'headers': {'Content-Type': 'application/json'}}
