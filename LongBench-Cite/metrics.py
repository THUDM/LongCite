import re
import string

import jieba
from fuzzywuzzy import fuzz
import difflib

from typing import List
from collections import Counter
from rouge import Rouge
import time, os, json
import requests


def gpt_score_qa(prediction, ground_truth, **kwargs):
    question = kwargs["query"]
    prompt = """You are asked to evaluate the quality of the AI assistant's answer to user questions as an impartial judge, and your evaluation should take into account factors including correctness (high priority), and comprehensiveness (whether the assistant's answer covers all points).\nRead the AI assistant's answer and compare against the reference answer, and give an overall integer rating in 1, 2, 3 (1 = wrong or irrelevant, 2 = partially correct, 3 = correct and comprehensive) based on the above principles, strictly in the following format:"[[rating]]", e.g. "[[2]]".\n\n[Question]\n$Q$\n[Reference answer]\n$A$\n[Assistant's answer]\n$P$\nRating:"""
    prompt = prompt.replace("$Q$", question).replace("$A$", ground_truth).replace("$P$", prediction)
    trys = 0
    score = "none"
    while (score == "none") and (trys < 5):
        response = query_gpt4(prompt)
        if isinstance(response, str) and 'Trigger' in response:
            prompt_tokens, completion_tokens = 0, 0
            score = "none"
            break
        try:
            prompt_tokens, completion_tokens = response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
            response = response["choices"][0]["message"]["content"]
            score = re.findall(r"\[\[([^\]]+)\]\]", response)[-1]
            matches = re.findall(r"\d+\.\d+|\d+", score)
            score = matches[0]
        except:
            trys += 1
            response = None
            prompt_tokens, completion_tokens = 0, 0
            score = "none"
    if score == "none":
        return 0.5
    kwargs["gpt_usage"]["responses"].append(response)
    kwargs["gpt_usage"]["prompt_tokens"] += prompt_tokens
    kwargs["gpt_usage"]["completion_tokens"] += completion_tokens
    return (int(score)-1.)/2

def gpt_score_summ(prediction, ground_truth, **kwargs):
    prompt = """You are asked to evaluate the quality of the AI assistant's generated summary as an impartial judge, and your evaluation should take into account factors including correctness (high priority), comprehensiveness (whether the assistant's summary covers all points), and coherence.\nRead the AI assistant's summary and compare against the reference summary, and give an overall integer rating in on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the evaluation criteria, strictly in the following format:"[[rating]]", e.g. "[[3]]".\n\n[Reference summary]\n$A$\n[Assistant's summary]\n$P$\nRating:"""
    prompt = prompt.replace("$A$", ground_truth).replace("$P$", prediction)
    trys = 0
    score = "none"
    while (score == "none") and (trys < 5):
        response = query_gpt4(prompt)
        if isinstance(response, str) and 'Trigger' in response:
            prompt_tokens, completion_tokens = 0, 0
            score = "none"
            break
        try:
            prompt_tokens, completion_tokens = response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"]
            response = response["choices"][0]["message"]["content"]
            score = re.findall(r"\[\[([^\]]+)\]\]", response)[-1]
            matches = re.findall(r"\d+\.\d+|\d+", score)
            score = matches[0]
        except:
            trys += 1
            response = None
            prompt_tokens, completion_tokens = 0, 0
            score = "none"
    if score == "none":
        return 0.5
    kwargs["gpt_usage"]["responses"].append(response)
    kwargs["gpt_usage"]["prompt_tokens"] += prompt_tokens
    kwargs["gpt_usage"]["completion_tokens"] += completion_tokens
    return (int(score)-1.)/4

GPT_MODEL = 'gpt-4o-2024-05-13'
def query_gpt4(prompt):
    tries = 0
    msg = [{"role": "system", "content": "You're a good assistant at evaluating the quality of texts."}, {"role": "user", "content": prompt}]
    while tries < 5:
        tries += 1
        try:
            headers = {
                'Authorization': "Bearer {Your Openai Key here}"
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json = {
                "model": GPT_MODEL,
                "messages": msg,
                "temperature": 0
            }, headers=headers, timeout=120)
            # print(tries)
            # print(resp.text)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            if "maximum context length" in str(e):
                raise e
            elif "triggering" in str(e):
                print(prompt)
                return 'Trigger Azure OpenAI\'s content management policy.'
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return
    
    return resp
