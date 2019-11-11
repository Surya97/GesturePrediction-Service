import json
import datetime


def handler(event, context):
    body = {}
    if event is not None and event['httpMethod'] == 'POST':
        body = json.loads(event['body'])

    data = {
        'output': body,
        'timestamp': datetime.datetime.utcnow().isoformat()
    }
    return {'statusCode': 200,
            'body': json.dumps(data),
            'headers': {'Content-Type': 'application/json'}}
