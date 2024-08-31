import requests
import time
import os, json
from tqdm import tqdm
import traceback

API_KEYS = {
    "openai": '',
    "zhipu": '',
    "anthropic": '',
    'vllm': 'token-abc123',
}

API_URLS = {
    'openai': 'https://api.openai.com/v1/chat/completions',
    'zhipu': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    "anthropic": 'https://api.anthropic.com/v1/messages',
    'vllm': 'http://127.0.0.1:8000/v1/chat/completions',
}

def query_llm(messages, model, temperature=1.0, max_new_tokens=1024, stop=None, return_usage=False):
    if 'gpt' in model:
        api_key, api_url = API_KEYS['openai'], API_URLS['openai']
    elif 'glm' in model:
        api_key, api_url = API_KEYS['zhipu'], API_URLS['zhipu']
    elif 'claude' in model:
        api_key, api_url = API_KEYS['anthropic'], API_URLS['anthropic']
    else:
        api_key, api_url = API_KEYS['vllm'], API_URLS['vllm']
    tries = 0
    while tries < 5:
        tries += 1
        try:
            if 'claude' not in model:
                headers = {
                    'Authorization': "Bearer {}".format(api_key),
                }
            else:
                headers = {
                    'x-api-key': api_key,
                    'anthropic-version': "2023-06-01",
                }
                
            resp = requests.post(api_url, json = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stop" if 'claude' not in model else 'stop_sequences': stop,
            }, headers=headers, timeout=600)
            # print(resp.text)
            # print(resp.status_code)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            elif "triggering" in str(e):
                return 'Trigger OpenAI\'s content management policy.'
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return None
    try:
        if 'content' not in resp["choices"][0]["message"] and 'content_filter_results' in resp["choices"][0]:
            resp["choices"][0]["message"]["content"] = 'Trigger OpenAI\'s content management policy.'
        if return_usage:
            return resp["choices"][0]["message"]["content"], resp['usage']
        else:
            return resp["choices"][0]["message"]["content"]
    except: 
        return None

if __name__ == '__main__':
    model = 'glm-4-0520'
    prompt = '你是谁'
    msg = [{'role': 'user', 'content': prompt}]
    output = query_llm(msg, model=model, temperature=1, max_new_tokens=10, stop=None, return_usage=True)
    print(output)