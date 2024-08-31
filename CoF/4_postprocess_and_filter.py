import os, json, re, jsonlines
from tqdm import tqdm
import traceback
import sys
sys.path.append('..')
from utils.retrieve import text_split_by_punctuation
from multiprocessing import Pool

ipt_file = './results/3_sentence_level_citation.jsonl'
fout_path = f"./results/4_final_data.jsonl"
os.system(f"rm -rf {fout_path}")

def strip(s):
    if s.strip() == "":
        return s
    if s.startswith('\n'):
        s = s[1:]
    if s.endswith('\n'):
        s = s[:-1]
    return s

def get_cite_str(c_js):
    if c_js['citation'] == []:
        return ""
    assert isinstance(c_js['citation'], str), c_js['citation']
    merged_citations = []
    sents = c_js['chunk_sents']
    spans = re.findall(r"\[([0-9]+\-[0-9]+)\]", c_js['citation'], re.DOTALL)
    for i, s in enumerate(spans):
        try:
            st, ed = [int(x) for x in s.split('-')]
            st, ed = max(0, st), min(ed, len(sents)-1)
            assert st <= ed
            merged_citations.append(f"[{sents[st]['c_idx']}-{sents[ed]['c_idx']}]")
        except:
            print(c_js['citation'].replace('\n', ''), len(sents), c_js['statement'])
            pass
    return "".join(merged_citations[:3])

def get_citations(statement, c_js):
    st, ed = statement.find("<cite>"), statement.find("</cite>")
    if st == -1:
        st = len(statement)
    res_statement = statement[:st]
    if c_js['statement'] not in res_statement:
        print(c_js['statement'] + '\n' + "*"*20 + '\n' + res_statement)
        raise NotImplementedError
    cite = get_cite_str(c_js)
    return strip(res_statement), cite
    
def process(js):
    try:
        answer, context = js['cited_answer'], js['context']
        all_c_sents = text_split_by_punctuation(context, return_dict=True)
        
        new_passage = ""
        new_passage2 = ""
        for i, c in enumerate(all_c_sents):
            st, ed = c['start_idx'], c['end_idx']
            assert c['content'] == context[st:ed], c
            ed = all_c_sents[i+1]['start_idx'] if i < len(all_c_sents)-1 else len(context)
            new_passage += f"<C{i}>"+context[st:ed]
        new_passage = '''Please answer the user's question based on the following document. When a sentence S in your response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the user's question.\n\n[Document Start]\n%s\n[Document End]''' % new_passage
            
        pos = 0
        statements = []
        i = 0
        while True:
            st = answer.find("<statement>", pos)
            if st == -1:
                st = len(answer)
            ed = answer.find("</statement>", st)
            statement = answer[pos:st]
            if len(statement.strip()) > 5:
                i += 1
            statements.append((statement, ""))
            if ed == -1:
                break
            assert ed > st, answer
            statement = answer[st+len("<statement>"):ed]
            if len(statement.strip()) > 0:
                statement, cite_str = get_citations(statement, js['statements'][i])
                statements.append((statement, cite_str))
                i += 1
            else:
                statements.append((statement, ""))
            pos = ed + len("</statement>")
        
        new_answer = ""
        cnt_s, cnt_c = 0, 0
        for statement, cite_str in statements:
            if statement.strip() == "":
                new_answer += statement
            else:
                new_answer += "<statement>{}<cite>{}</cite></statement>".format(statement, cite_str)
                cnt_s += 1
                cnt_c += (cite_str.strip() != "")

        if cnt_c == 0 or (cnt_s >= 5 and cnt_c/cnt_s < 0.2):
            print(new_answer)
            print("Too few cite", cnt_c, cnt_s)
            return 1

        res = [{"role": "user", "content": new_passage + '\n\n' + js['query']}, {"role": "assistant", "content": new_answer}]
            
        with open(fout_path, "a") as fout:
            fout.write(json.dumps(res, ensure_ascii=False)+'\n')
            fout.flush()
        return 0
    except:
        print('\n\n')
        traceback.print_exc()
        print(js['idx'])
        print('-'*200)
        return 1

need_list = [x for x in tqdm(jsonlines.open(ipt_file, "r"))]
with Pool(32) as p:
    rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))
num_bad_cases = sum(rst)
print(f'Total: {len(need_list)} | Bad Cases: {num_bad_cases} | Remain: {len(need_list)-num_bad_cases}')
