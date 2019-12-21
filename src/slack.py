import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL')
assert WEBHOOK_URL is not None


def slack_message(text: str):
    message = {'text': text}
    response = requests.post(
        WEBHOOK_URL, data=json.dumps(message),
        headers={'Content-Type': 'application/json'}
    )
    return response


if __name__ == '__main__':
    response = slack_message('hello world!')
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'
            % (response.status_code, response.text)
        )
