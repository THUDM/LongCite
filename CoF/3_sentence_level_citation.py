import os, sys
sys.path.append('../')
from tqdm import tqdm
import json, jsonlines
from utils.retrieve import text_split_by_punctuation, text_split, cat_chunks
from utils.llm_api import query_llm
from multiprocessing import Pool
import multiprocessing
from functools import partial
import traceback

cite_model = "glm-4-0520"
parallel_num = 1
ipt_path = "./results/2_chunk_level_citation.jsonl"
fout_path = "./results/3_sentence_level_citation.jsonl"

with open('prompt_sentence_level_citation.txt', "r") as fp:
    prompt_format = fp.read()
context_chunk_size = 128

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

def process(js):
    try:
        idx, query = js['idx'], js['query']
        context = js['context']
        all_c_chunks = text_split(context, chunk_size=context_chunk_size, return_token_ids=True)
        all_c_sents = text_split_by_punctuation(context, return_dict=True)
        for i, c_js in enumerate(js['statements']):
            citations = sorted(c_js['citation'], key=lambda x:x['start'])
            if len(citations) == 0:
                continue
            statement = c_js['statement']
            spans = [(max(0, x['start']//context_chunk_size-1), min(len(all_c_chunks), x['start']//context_chunk_size+2)) for x in citations]
            relevant_chunks = []
            context_sents = []
            st, ed, j, sidx = spans[0][0], spans[0][1], 0, 0
            while j < len(citations):
                if j == len(citations)-1:
                    passage = cat_chunks(all_c_chunks[st:ed])
                    for rm in range(5):
                        pos = context.find(passage)
                        if pos != -1:
                            break
                        passage = passage[1:-1]
                    if pos == -1:
                        for rm in range(1, 5):
                            passage = cat_chunks(all_c_chunks[st:ed], remove_head_tail=rm)
                            pos = context.find(passage)
                            if pos != -1:
                                break

                    assert pos != -1, passage
                    sents = []
                    for s in all_c_sents:
                        if (s['start_idx'] >= pos and s['start_idx'] <= pos + len(passage)) or (s['end_idx'] >= pos and s['end_idx'] <= pos + len(passage)):
                            sents.append(s) 
                    new_passage = ""
                    for k, s in enumerate(sents):
                        sst, sed = s['start_idx'], s['end_idx']
                        assert s['content'] == context[sst:sed], s
                        sed = sents[k+1]['start_idx'] if k < len(sents)-1 else s['end_idx']
                        new_passage += f"<C{sidx}>"+context[sst:sed]
                        context_sents.append({
                            'content': context[sst:sed],
                            'start': sst,
                            'end': sed,
                            'cur_idx': sidx, 
                            'c_idx': s['c_idx'],
                        })
                        sidx += 1
                    relevant_chunks.append(new_passage)
                    break
                j = j + 1
                if spans[j][0] > ed:
                    passage = cat_chunks(all_c_chunks[st:ed])
                    for rm in range(5):
                        pos = context.find(passage)
                        if pos != -1:
                            break
                        passage = passage[1:-1]
                    if pos == -1:
                        for rm in range(1, 5):
                            passage = cat_chunks(all_c_chunks[st:ed], remove_head_tail=rm)
                            pos = context.find(passage)
                            if pos != -1:
                                break
                    assert pos != -1, passage
                    sents = []
                    for s in all_c_sents:
                        if (s['start_idx'] >= pos and s['start_idx'] <= pos + len(passage)) or (s['end_idx'] >= pos and s['end_idx'] <= pos + len(passage)):
                            sents.append(s) 
                    new_passage = ""
                    for k, s in enumerate(sents):
                        sst, sed = s['start_idx'], s['end_idx']
                        assert s['content'] == context[sst:sed], s
                        sed = sents[k+1]['start_idx'] if k < len(sents)-1 else s['end_idx']
                        new_passage += f"<C{sidx}>"+context[sst:sed]
                        context_sents.append({
                            'content': context[sst:sed],
                            'start': sst,
                            'end': sed,
                            'cur_idx': sidx, 
                            'c_idx': s['c_idx'],
                        })
                        sidx += 1
                    relevant_chunks.append(new_passage)
                    st, ed = spans[j]
                else:
                    ed = spans[j][1]
            context_str = '\n\n'.join(relevant_chunks)
            prompt = prompt_format.replace('<<context>>', context_str).replace('<<question>>', query).replace('<<statement>>', statement)
            print(prompt)
            msg = [{'role': 'user', 'content': prompt}]
            output = query_llm(msg, model=cite_model, temperature=1, max_new_tokens=128)
            print(output)
            if output is None:
                return 1
            js['statements'][i]['expand_chunk'] = context_str
            js['statements'][i]['chunk_sents'] = context_sents
            js['statements'][i]['citation'] = output
            
        with open(fout_path, "a") as fout:
            fout.write(json.dumps(js, ensure_ascii=False)+'\n')
            fout.flush()
        # json.dump(js, open("test2.json", 'w'), indent=2, ensure_ascii=False)
        return 0
    except:
        print(idx)
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
    