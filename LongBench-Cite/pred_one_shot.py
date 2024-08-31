import os, sys
from tqdm import tqdm
import json, jsonlines
from multiprocessing import Pool
import traceback
from nltk.tokenize import PunktSentenceTokenizer
import re
sys.path.append('../')
from utils.llm_api import query_llm
from utils.retrieve import  text_split_by_punctuation

cite_model = "gpt-4o-2024-05-13" # model name or the path to open-sourced model.
ipts = json.load(open("LongBench-Cite.json"))
save_name = cite_model.split('/')[-1]
save_dir = 'preds'
os.makedirs(f'{save_dir}/tmp', exist_ok=True)
fout_path = f'{save_dir}/tmp/{save_name}.jsonl'
if os.path.exists(fout_path):
    with jsonlines.open(fout_path, 'r') as f:
        opts = [x for x in f]
else:
    opts = []
s = set(x['idx'] for x in opts)
need_list = [x for x in ipts if x['idx'] not in s][:1]
print(f'Model: {cite_model}')
print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')
if len(need_list) == 0:
    exit()

with open('one_shot_prompt.txt', "r") as fp:
    prompt_format = fp.read()

def get_citations(statement, sents):
    c_texts = re.findall(r'<cite>(.*?)</cite>', statement, re.DOTALL)
    spans = sum([re.findall(r"\[([0-9]+\-[0-9]+)\]", c_text, re.DOTALL) for c_text in c_texts], [])
    statement = re.sub(r'<cite>(.*?)</cite>', '', statement, flags=re.DOTALL)
    merged_citations = []
    for i, s in enumerate(spans):
        try:
            st, ed = [int(x) for x in s.split('-')]
            if st > len(sents) - 1 or ed < st:
                continue
            st, ed = max(0, st), min(ed, len(sents)-1)
            assert st <= ed, str(c_texts) + '\t' + str(len(sents))
            if len(merged_citations) > 0 and st == merged_citations[-1]['ed_sent'] + 1:
                merged_citations[-1].update({
                    "ed_sent": ed,
                    'end_char': sents[ed]['end'],
                    'cite': ''.join([x['content'] for x in sents[merged_citations[-1]['st_sent']:ed+1]]),
                })
            else:
                merged_citations.append({
                    "st_sent": st,
                    "ed_sent": ed,
                    "start_char":  sents[st]['start'],
                    'end_char': sents[ed]['end'],
                    'cite': ''.join([x['content'] for x in sents[st:ed+1]]),
                })
        except:
            print(c_texts, len(sents), statement)
            raise
    return statement, merged_citations[:3]

def postprocess(answer, context, sents):
    chunks = []
    for k, s in enumerate(sents):
        sst, sed = s['start_idx'], s['end_idx']
        assert s['content'] == context[sst:sed], s
        sed = sents[k+1]['start_idx'] if k < len(sents)-1 else s['end_idx']
        chunks.append({
            'content': context[sst:sed],
            'start': sst,
            'end': sed,
            'c_idx': s['c_idx'],
        })
    
    res = []
    pos = 0
    while True:
        st = answer.find("<statement>", pos)
        if st == -1:
            st = len(answer)
        ed = answer.find("</statement>", st)
        statement = answer[pos:st]
        if len(statement.strip()) > 5:
            res.append({
                "statement": statement.strip(),
                "citation": []
            })
        
        if ed == -1:
            break

        statement = answer[st+len("<statement>"):ed]
        if len(statement.strip()) > 0:
            statement, citations = get_citations(statement, chunks)
            res.append({
                "statement": statement.strip(),
                "citation": citations
            })
        pos = ed + len("</statement>")
    return res
            
def process(js):
    try:
        context, query = js['context'], js['query']
        sents = text_split_by_punctuation(context, return_dict=True)

        passage = ""
        for i, c in enumerate(sents):
            st, ed = c['start_idx'], c['end_idx']
            assert c['content'] == context[st:ed], c
            ed = sents[i+1]['start_idx'] if i < len(sents)-1 else len(context)
            passage += f"<C{i}>"+context[st:ed]
        
        prompt = prompt_format.replace('<<context>>', passage).replace('<<question>>', query)
        # print(prompt)
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=cite_model, temperature=1, max_new_tokens=1024)
        if output is None:
            raise NotImplementedError("Error: No Output")
        if output.startswith("statement>"):
            output = "<" + output
        # print(output)
        statements_with_citations = postprocess(output, context, sents)
        res = {
            'idx': js['idx'],
            'dataset': js['dataset'],
            'query': js['query'],
            'answer': js['answer'],
            'few_shot_scores': js['few_shot_scores'],
            'prediction': output,
            'statements': statements_with_citations,
        }
        with open(fout_path, "a") as fout:
            fout.write(json.dumps(res, ensure_ascii=False)+'\n')
            fout.flush()
        return res
    except:
        print(js['idx'])
        print(query)
        traceback.print_exc()
        print('-'*200)
        return None

with Pool(4) as p:
    rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))
rst = opts + [x for x in rst if x is not None]
rst = sorted(rst, key=lambda x:x['idx'])
json.dump(rst, open(f'{save_dir}/{save_name}.json', "w"), indent=2, ensure_ascii=False)
    
    
