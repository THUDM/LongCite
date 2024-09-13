<p align="center" width="100%">
<img src="https://github.com/user-attachments/assets/089ccebe-2ebc-4c8e-a9e0-9d82f5e253f9" width="80%" alt="LongCite"> 
</p>

# LongCite: 让LLM在长上下文问答中生成细粒度引用

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THUDM/LongCite-45k" target="_blank">HF 库</a> • 📃 <a href="https://arxiv.org/abs/2409.02897" target="_blank">论文</a>  • 🚀 <a href="https://huggingface.co/spaces/THUDM/LongCite" target="_blank">HF空间</a>
</p>

[English](./README.md) | [中文](./README_zh.md) | [日本語](./README_ja.md)

https://github.com/user-attachments/assets/474b3190-d0a2-4947-920a-445dd9aff217


## 🔍 目录
- [⚙️ LongCite 部署](#部署)
- [🤖️ 数据构造流程 CoF](#流水线)
- [🖥️ 模型训练](#训练)
- [📊 评估](#评估)
- [📝 引用](#引用)

<a name="部署"></a>
## ⚙️ LongCite 部署

**环境设置**:
我们建议使用 `transformers>=4.43.0` 来成功部署我们的模型。

我们开源了两个模型: [LongCite-glm4-9b](https://huggingface.co/THUDM/LongCite-glm4-9b) 和 [LongCite-llama3.1-8b](https://huggingface.co/THUDM/LongCite-llama3.1-8b)。这两个模型分别基于 [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b) 和 [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) 训练，支持最大 128K 上下文。它们对应论文中的 "LongCite-9B" 和 "LongCite-8B" 模型。给定一个长上下文的问题，这些模型可以生成准确的回答和精确的句子级别引用，便于用户验证模型输出的信息。试用该模型:
```python
import json
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
你可以通过运行以下命令部署你自己的 LongCite 聊天机器人（如视频中演示的）：
```
CUDA_VISIBLE_DEVICES=0 streamlit run demo.py --server.fileWatcherType none
```
你也可以通过 [vllm](https://github.com/vllm-project/vllm) 部署模型，这可以使生成更快并支持多并发服务。代码示例在 [vllm_inference.py](https://github.com/THUDM/LongCite/blob/main/vllm_inference.py) 中。

<a name="流水线"></a>
## 🤖️ 数据构造流程 CoF
![cof](https://github.com/user-attachments/assets/dae25838-3ce0-4a2c-80f7-307c8128e5c4)

我们将 CoF（Corse to Fine）开源在 `CoF/` 目录下，这是我们的自动化SFT数据构建流程，用来生成高质量的的带有细粒度引用的长上下文问答数据。请在 `utils/llm_api.py` 中配置你的API密钥，然后运行以下四个脚本以获得最终数据：`1_qa_generation.py`，`2_chunk_level_citation.py`，`3_sentence_level_citation.py`，和 `4_postprocess_and_filter.py`。

<a name="训练"></a>

## 🖥️ 模型训练

你可以通过 Hugging Face 数据集 ([🤗 HF Repo](https://huggingface.co/datasets/THUDM/LongCite-45k)) 下载和保存 **LongCite-45k** 数据集：
```python
dataset = load_dataset('THUDM/LongCite-45k')
for split, split_dataset in dataset.items():
    split_dataset.to_json("train/long.jsonl")
```
你可以将其与一般的 SFT 数据如 [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main/HTML_cleaned_raw_dataset) 混合使用。我们采用 [Metragon-LM](https://github.com/NVIDIA/Megatron-LM) 进行模型训练。如果想采用更轻量级的实现，你可以采用 [LongAlign](https://github.com/THUDM/LongAlign) 的代码和环境，对于训练GLM-4-9B 和 Llama-3.1-8B，它可以支持 32k token的最大训练序列长度。

<a name="评估"></a>
## 📊 评估
我们引入了一个自动评估基准：**LongBench-Cite**，它采用了 [LongBench](https://github.com/THUDM/LongBench) 和 [LongBench-Chat](https://github.com/THUDM/LongAlign) 中的问答数据，用于衡量长上下文 QA 场景中的引用质量以及响应的正确性。 

我们在 `LongBench-Cite/` 目录下提供了评估数据和代码。运行 `pred_sft.py` 和 `pred_one_shot.py` 以从微调模型（如 LongCite-glm4-9b）和普通模型（如 GPT-4o）获取响应。然后运行 `eval_cite.py` 和 `eval_correct.py` 以评估引用质量和响应的正确性。请在 `utils/llm_api.py` 中配置你的 OpenAI API 密钥，因为我们采用 GPT-4o 作为评审。 

以下是 **LongBench-Cite** 的评估结果：
![eval_results](https://github.com/user-attachments/assets/1ef68c5f-63f0-4041-8f19-b8694b0d68c2)

<a name="引用"></a>
## 📝 引用

如果你觉得我们的工作有用，请考虑引用 LongCite：

```
@article{zhang2024longcite,
  title = {LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA} 
  author={Jiajie Zhang and Yushi Bai and Xin Lv and Wanjun Gu and Danqing Liu and Minhao Zou and Shulin Cao and Lei Hou and Yuxiao Dong and Ling Feng and Juanzi Li},
  journal={arXiv preprint arXiv:2409.02897},
  year={2024}
}
```
