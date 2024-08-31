<p align="center" width="100%">
<img src="https://github.com/user-attachments/assets/d931d4a7-fb5d-4b9c-af54-12bdc875f8e1" width="80%" alt="LongCite">
</p>

# LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THUDM/LongCite-45k" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/abs/" target="_blank">Paper</a>
</p>

[English](./README.md) | [中文](./README_zh.md)

https://github.com/user-attachments/assets/68f6677a-3ffd-41a8-889c-d56a65f9e3bb

## 🔍 Table of Contents
- [⚙️ LongCite Deployment](#deployment)
- [🤖️ CoF pipeline](#pipeline)
- [🖥️ Model Training](#training)
- [📊 Evaluation](#evaluation)
- [📝 Citation](#citation)

<a name="deployment"></a>
## ⚙️ LongCite Deployment

**Environmental Setup**:
We recommend using `transformers>=4.43.0` to successfully deploy our models.

We open-source two models: [LongCite-glm4-9b](https://huggingface.co/THUDM/LongCite-glm4-9b) and [LongCite-llama3.1-8b](https://huggingface.co/THUDM/LongCite-llama3.1-8b), trained based on [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b) and [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), respectively. These two models point to the "LongCite-9B" and "LongCite-8B" models in our paper. Given a long-context-based query, these models can generate accurate response and precise sentence-level citations, making it easy for user to verify the output information. Try the model:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('THUDM/LongCite-glm4-9b', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('THUDM/LongCite-glm4-9b', torch_dtype=torch.bfloat16, trust_remote_code=True, device_map='auto')

context = '''
W. Russell Todd, 94, United States Army general (b. 1928). February 13. Tim Aymar, 59, heavy metal singer (Pharaoh) (b. 1963). Marshall \"Eddie\" Conway, 76, Black Panther Party leader (b. 1946). Roger Bonk, 78, football player (North Dakota Fighting Sioux, Winnipeg Blue Bombers) (b. 1944). Conrad Dobler, 72, football player (St. Louis Cardinals, New Orleans Saints, Buffalo Bills) (b. 1950). Brian DuBois, 55, baseball player (Detroit Tigers) (b. 1967). Robert Geddes, 99, architect, dean of the Princeton University School of Architecture (1965–1982) (b. 1923). Tom Luddy, 79, film producer (Barfly, The Secret Garden), co-founder of the Telluride Film Festival (b. 1943). David Singmaster, 84, mathematician (b. 1938).
'''
query = "What was Robert Geddes' profession?"
result = model.query_longcite(context, query, tokenizer=tokenizer, max_input_length=128000, max_new_tokens=1024)

print("Answer:\n{}\n".format(result['answer']))
print("Statement with citations:\n{}\n".format(
  json.dumps(result['statements_with_citations'], indent=2, ensure_ascii=False)))
print("Context (divided into sentences):\n{}\n".format(result['splited_context']))
```
You may deploy your own LongCite chatbot (like the one we show in the above video) by running
```
CUDA_VISIBLE_DEVICES=0 streamlit run demo.py --server.fileWatcherType none
```
Alternatively, you can deploy the model with [vllm](https://github.com/vllm-project/vllm), which allows faster generation and multiconcurrent server. See the code example in [vllm_inference.py](https://github.com/THUDM/LongCite/blob/main/vllm_inference.py).

<a name="pipeline"></a>
## 🤖️ CoF Pipeline
![cof](https://github.com/user-attachments/assets/dae25838-3ce0-4a2c-80f7-307c8128e5c4)

We are also open-sourcing CoF (Coarse to Fine) under `CoF/`, our automated data construction pipeline for long-context QA with citations. Please configure your API key in the [utils/llm_api.py](https://github.com/THUDM/LongCite/blob/main/utils/llm_api.py), then run the following four scripts to obtain the final data: 
`1_qa_generation.py`, `2_chunk_level_citation.py`, `3_sentence_level_citaion.py`, and `4_postprocess_and_filter.py`.


<a name="training"></a>
## 🖥️ Model Training

You can download and save the **LongCite-45k** dataset through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/THUDM/LongCite-45k)):
```python
dataset = load_dataset('THUDM/LongCite-45k')
for split, split_dataset in dataset.items():
    split_dataset.to_json("train/LongCite-45k.jsonl")
```
You can mix it with general SFT data such as [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main/HTML_cleaned_raw_dataset). We adopt [Metragon-LM](https://github.com/NVIDIA/Megatron-LM) for model training. For a more lightweight implementation, you may adopt the code and environment from [LongAlign](https://github.com/THUDM/LongAlign), which can support a max training sequence length of 32k tokens for GLM-4-9B and Llama-3.1-8B. 

<a name="evaluation"></a>
## 📊 Evaluation
We introduce an automatic benchmark: **LongBench-Cite**, to measuring both the response correctness and citation quality in long-context QA scenarios. 
We provide our evaluation data and code under `LongBench-Cite/`. Run `pred_sft.py` and `pred_one_shot.py` to get responses from fine-tuned models (e.g., LongCite-glm4-9b) and normal models (e.g., GPT-4o). Then run `eval_correct.py` and `eval_cite.py` to evaluate the response correctness ($S_q$) and citation quality ($S_l$). Remember to configure your OpenAI API key in [utils/llm_api.py](https://github.com/THUDM/LongCite/blob/main/utils/llm_api.py) since we adopt GPT-4o as the judge.

Here are the evaluation results on **LongBench-Write**:
<img width="1000" alt="longbench-write" src="https://github.com/user-attachments/assets/8dbb6c02-09c4-4319-bd38-f1135457cd25">
Here are the evaluation results on **LongWrite-Ruler**:
![longwrite-ruler](https://github.com/user-attachments/assets/471f6e74-ab2c-4ad7-b73f-9ec8d2c2cde5)

<a name="citation"></a>
## 📝 Citation

If you find our work useful, please kindly cite:

```
@article{bai2024LongCite,
  title={LongCite: Unleashing 10,000+ Word Generation from Long Context LLMs}, 
  author={Yushi Bai and Jiajie Zhang and Xin Lv and Linzhi Zheng and Siqi Zhu and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  journal={arXiv preprint arXiv:2408.07055},
  year={2024}
}
```
