import os, sys
sys.path.append('../')
from tqdm import tqdm
import json, jsonlines
from utils.retrieve import batch_search, text_split, text_split_by_punctuation
from utils.llm_api import query_llm
from multiprocessing import Pool
import multiprocessing
import traceback
import re

cite_model = "glm-4-0520"
parallel_num = 1
ipt_path = "./results/1_qa.jsonl"
fout_path = "./results/2_chunk_level_citation.jsonl"

with open('prompt_chunk_level_citation.txt', "r") as fp:
    prompt_format = fp.read()
context_chunk_size = 128
top_retrieval_k = 10
max_retrieval_tot = 40

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass

import multiprocessing.pool
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc
    
def run_retrieval(answer, context, chunk_size=128):
    statements = text_split_by_punctuation(answer)
    all_c_chunks = text_split(context, chunk_size=chunk_size)
    output = batch_search(queries=statements, contexts=all_c_chunks, k=top_retrieval_k)
    k = min(top_retrieval_k, (max_retrieval_tot + len(statements) - 1) // len(statements))
    topk_c_chunks = []
    s = set()
    for chunks in output:
        for c in chunks['retrieve_results'][:k]:
            if c['content'] not in s:
                topk_c_chunks.append(c)
                s.add(c['content'])
    topk_c_chunks = sorted(topk_c_chunks, key=lambda x:x['start_idx'])
    return topk_c_chunks, all_c_chunks

def get_citations(statement, chunks):
    st, ed = statement.find("<cite>"), statement.find("</cite>")
    if st == -1:
        st = len(statement)
    statement, c_text = statement[:st].strip(), statement[st:]
    c_idxs = re.findall(r"\[([0-9]+)\]", c_text, re.DOTALL)
    citations = []
    for idx in c_idxs:
        idx = int(idx) - 1
        if idx >= 0 and idx < len(chunks):
            citations.append({
                'start': chunks[idx]['start_idx'],
                'end': chunks[idx]['end_idx'],
                'idx': idx+1,
                'cite': chunks[idx]['content']
            })
        if len(citations) == 5:
            break
    return statement, citations
    
def postprocess(answer, chunks):
    try:
        pos = 0
        res = []
        while True:
            st = answer.find("<statement>", pos)
            if st == -1:
                st = len(answer)
            ed = answer.find("</statement>", st)
            statement = answer[pos:st].strip()
            if len(statement) > 5:
                res.append({
                    "statement": statement,
                    "citation": []
                })
            if ed == -1:
                break
            assert ed > st, answer
            statement = answer[st+len("<statement>"):ed]
            if len(statement.strip()) > 0:
                statement, citations = get_citations(statement, chunks)
                res.append({
                    "statement": statement,
                    "citation": citations
                })
            pos = ed + len("</statement>")
        return res
    except:
        traceback.print_exc()
        print("-"*200)
        return None

def process(js):
    try:
        idx = js['idx']
        context = js['context']
        query, answer = js['query'], js['answer']
        topk_c_chunks, all_c_chunks = run_retrieval(answer, context, chunk_size=context_chunk_size)
        
        context_str = []
        for i, c in enumerate(topk_c_chunks):
            c['top_idx'] = i+1
            context_str.append(f"Snippet [{i+1}]\n{c['content'].strip()}")
        context_str = '\n\n'.join(context_str)
        
        prompt = prompt_format.replace('<<context>>', context_str).replace('<<question>>', query).replace('<<answer>>', answer)
        # print(prompt)
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=cite_model, temperature=1, max_new_tokens=2048)
        # print(output)
        if output is None:
            return 1
        cited_answer = output.split('[Answer with Citations]')[-1]
        if cited_answer.startswith("statement>"):
            cited_answer = "<"+cited_answer
        cited_answer = cited_answer.replace("<statement ", "<statement>").replace("<statement,", "<statement>").replace("<statement><statement>", "<statement>").replace("</statement></statement>", "</statement>").replace("</statement><cite></cite></statement>", "<cite></cite></statement>").replace("< cite>", "<cite>")
        statements_with_citations = postprocess(cited_answer, topk_c_chunks)
        if statements_with_citations is None:
            return 1
        res = {
            'idx': idx,
            'query': query,
            'answer': answer,
            'cited_answer': cited_answer,
            'statements': statements_with_citations,
            'context': context
        }
        with open(fout_path, "a") as fout:
            fout.write(json.dumps(res, ensure_ascii=False)+'\n')
            fout.flush()
        # json.dump(res, open("test.json", 'w'), indent=2, ensure_ascii=False)
        return 0
    except:
        print(id)
        print(query)
        traceback.print_exc()
        print('-'*200)
        return 1

if __name__ == '__main__':
    
    if os.path.exists(fout_path):
        with jsonlines.open(fout_path, 'r') as f:
            opts = [x for x in tqdm(f)]
    else:
        opts = []
    s = set(x['idx'] for x in opts)
    ipts = [x for x in tqdm(jsonlines.open(ipt_path, "r"))]
    need_list = [x for x in ipts if x['idx'] not in s]
    print(f'Already process: {len(opts)} | Remain to process: {len(need_list)}')
    with NoDaemonProcessPool(parallel_num) as p:
        rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))
    num_bad_cases = sum(rst)
    print(f'There are {num_bad_cases} bad cases. You can run this scripts again to re-process these bad cases.')
    
    