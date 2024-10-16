import os, json, jsonlines
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback
import torch.multiprocessing as mp
from collections import defaultdict
import time

gpus = [0,1,2,3,4,5,6,7]
ipts = json.load(open("LongBench-Cite.json"))
model_path = "THUDM/LongCite-glm4-9b" # THUDM/LongCite-llama3.1-8b
save_name = model_path.split('/')[-1]
save_dir = 'preds'
os.makedirs(f'{save_dir}/tmp', exist_ok=True)
fout_path = f'{save_dir}/tmp/{save_name}.jsonl'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

for gpu_per_model in [1, 8]:
    parallel_num = len(gpus) // gpu_per_model
    if os.path.exists(fout_path):
        with jsonlines.open(fout_path, 'r') as f:
            opts = [x for x in f]
    else:
        opts = []
    s = set(x['idx'] for x in opts)
    need_list = [x for x in ipts if x['idx'] not in s]
    print(f'Model: {model_path} | GPU per model: {gpu_per_model} | parallel num: {parallel_num}')
    print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')
    if len(need_list) == 0:
        break

    def get_pred(rank, data):
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpus[rank*gpu_per_model:(rank+1)*gpu_per_model])
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        for js in tqdm(data):
            try:
                context, query = js['context'], js['query']
                res = model.query_longcite(context, query, tokenizer=tokenizer, max_input_length=128000, max_new_tokens=1024)
                res = {
                    'idx': js['idx'],
                    'dataset': js['dataset'],
                    'query': js['query'],
                    'answer': js['answer'],
                    'few_shot_scores': js['few_shot_scores'],
                    'prediction': res['answer'],
                    'statements': res['statements_with_citations'],
                }
                with open(fout_path, "a") as fout:
                    fout.write(json.dumps(res, ensure_ascii=False)+'\n')
                    fout.flush()
            except KeyboardInterrupt as e:
                raise e
            except:
                print(js['idx'])
                print(query)
                traceback.print_exc()
                print('-'*200)
        del model

    need_list_subsets = [need_list[i::parallel_num] for i in range(parallel_num)]
    processes = []
    for rank in range(parallel_num):
        p = mp.Process(target=get_pred, args=(rank, need_list_subsets[rank]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

with jsonlines.open(fout_path, 'r') as f:
    opts = sorted([x for x in f], key=lambda x:x['idx'])
json.dump(opts, open(f'{save_dir}/{save_name}.json', 'w'), indent=2, ensure_ascii=False)
