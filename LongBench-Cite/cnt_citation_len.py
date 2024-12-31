import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)

paths = [
    # "./scores_cite/LongCite-glm4-9b.json"
]

for path in paths:
    print()
    print(path)
    total_citation_len, citation_num = 0, 0
    data = json.load(open(path))
    for js in tqdm(data): 
        if "finish" in js:
            continue
        for sc in js['statements']:
            num_statement += 1
            for c in sc['citation']:
                cite_len = len(tokenizer.encode(c['cite'], add_special_tokens=False))
                total_citation_len += cite_len
                citation_num += 1
    print(total_citation_len / citation_num)

