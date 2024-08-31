<p align="center" width="100%">
<img src="https://github.com/user-attachments/assets/d931d4a7-fb5d-4b9c-af54-12bdc875f8e1" width="80%" alt="longwriter">
</p>

# LongWriter: Unleashing 10,000+ Word Generation From Long Context LLMs

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THUDM/LongWriter-6k" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/abs/2408.07055" target="_blank">Paper</a> • 🚀 <a href="https://huggingface.co/spaces/THUDM/LongWriter" target="_blank">HF Space</a>
</p>

[English](./README.md) | [中文](./README_zh.md) | [日本語](./README_jp.md)

https://github.com/user-attachments/assets/c7eedeca-98ed-43ec-8619-25137987bcde

Left: LongWriter-glm4-9b; Right: GLM-4-9B-chat

## 🔥 Updates
**[2024/08/18]** You can now deploy the LongWriter model using [vllm](https://github.com/vllm-project/vllm). Refer to the code in [vllm_inference.py](https://github.com/THUDM/LongWriter/blob/main/vllm_inference.py) and experience lightning-fast generation. It can **generate over 10,000+ words in just one minute**!

## 🔍 Table of Contents
- [⚙️ LongWriter Deployment](#deployment)
- [🤖️ AgentWrite](#agentwrite)
- [🖥️ Model Training](#longwriter-training)
- [📊 Evaluation](#evaluation)
- [👀 Cases](#case)
- [📝 Citation](#citation)

<a name="deployment"></a>
## ⚙️ LongWriter Deployment

**Environmental Setup**:
We recommend using `transformers>=4.43.0` to successfully deploy our models.

We open-source two models: [LongWriter-glm4-9b](https://huggingface.co/THUDM/LongWriter-glm4-9b) and [LongWriter-llama3.1-8b](https://huggingface.co/THUDM/LongWriter-llama3.1-8b), trained based on [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b) and [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), respectively. These two models point to the "LongWriter-9B-DPO" and "LongWriter-8B" models in our paper. Try the model:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("THUDM/LongWriter-glm4-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("THUDM/LongWriter-glm4-9b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
model = model.eval()
query = "Write a 10000-word China travel guide"
response, history = model.chat(tokenizer, query, history=[], max_new_tokens=32768, temperature=0.5)
print(response)
```
You may deploy your own LongWriter chatbot (like the one we show in the teasor video) by running
```
CUDA_VISIBLE_DEVICES=0 python trans_web_demo.py
```
Alternatively, you can deploy the model with [vllm](https://github.com/vllm-project/vllm), which allows generating 10,000+ words within a minute! See the code example in [vllm_inference.py](https://github.com/THUDM/LongWriter/blob/main/vllm_inference.py).

<a name="agentwrite"></a>
## 🤖️ AgentWrite

![agentwrite](https://github.com/user-attachments/assets/5d80314b-eab6-4945-848d-0db8e23ffc90)

We are also open-sourcing AgentWrite under `agentwrite/`, our automated ultra-long output data construction pipeline. Run `plan.py` and then `write.py` to obtain the final data. Please configure your API key in the files.


<a name="longwriter-training"></a>
## 🖥️ Model Training

You can download and save the **LongWriter-6k** data through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/THUDM/LongWriter-6k)):
```python
dataset = load_dataset('THUDM/LongWriter-6k')
for split, split_dataset in dataset.items():
    split_dataset.to_json("train/LongWriter-6k.jsonl")
```
You can mix it with your own general SFT data. We adopt the code and environment in [LongAlign](https://github.com/THUDM/LongAlign) for model training (we use `transformers==4.43.0` for training on Llama-3.1), with slight modification to adapt to new models. The training code is under `train/`. Please make sure to install FlashAttention 2 according to the code base of [FlashAttention](https://github.com/Dao-AILab/flash-attention).

<a name="evaluation"></a>
## 📊 Evaluation
We introduce two evaluation benchmarks: **LongBench-Write** and **LongWrite-Ruler**. **LongBench-Write** focuses more on measuring the long output quality as well as the output length, while **LongWrite-Ruler** is designed as a light-weight stress test of the model's maximum output length.
We provide our evaluation data and code under `evaluation/`. Run
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python pred.py
```
to get model responses. Then run `python eval_quality.py` and `python eval_length.py` to evaluate the quality ($S_q$) and length ($S_l$) scores. Remember to configure your OpenAI API key in `eval_quality.py` since we adopt GPT-4o as the judge.

Here are the evaluation results on **LongBench-Write**:
<img width="1000" alt="longbench-write" src="https://github.com/user-attachments/assets/8dbb6c02-09c4-4319-bd38-f1135457cd25">
Here are the evaluation results on **LongWrite-Ruler**:
![longwrite-ruler](https://github.com/user-attachments/assets/471f6e74-ab2c-4ad7-b73f-9ec8d2c2cde5)


<a name="case"></a>
## 👀 Cases
Here are LongWriter-glm4-9b's outputs to random test prompts.

*User: Write a tragic love story about a lord's daughter falling in love with a servant, 5000 words.*

<a name="citation"></a>
## 📝 Citation

If you find our work useful, please kindly cite:

```
@article{bai2024longwriter,
  title={LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs}, 
  author={Yushi Bai and Jiajie Zhang and Xin Lv and Linzhi Zheng and Siqi Zhu and Lei Hou and Yuxiao Dong and Jie Tang and Juanzi Li},
  journal={arXiv preprint arXiv:2408.07055},
  year={2024}
}
```