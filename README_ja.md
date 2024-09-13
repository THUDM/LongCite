<p align="center" width="100%">
<img src="https://github.com/user-attachments/assets/089ccebe-2ebc-4c8e-a9e0-9d82f5e253f9" width="80%" alt="LongCite"> 
</p>

# LongCite: 長文コンテキストQAにおける細粒度引用を生成するLLM

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THUDM/LongCite-45k" target="_blank">HF リポジトリ</a> • 📃 <a href="https://arxiv.org/abs/2409.02897" target="_blank">論文</a>  • 🚀 <a href="https://huggingface.co/spaces/THUDM/LongCite" target="_blank">HFスペース</a>
</p>

[English](./README.md) | [中文](./README_zh.md) | 日本語

https://github.com/user-attachments/assets/474b3190-d0a2-4947-920a-445dd9aff217


## 🔍 目次
- [⚙️ LongCite デプロイ](#デプロイ)
- [🤖️ データ構築パイプライン CoF](#パイプライン)
- [🖥️ モデルトレーニング](#トレーニング)
- [📊 評価](#評価)
- [📝 引用](#引用)

<a name="デプロイ"></a>
## ⚙️ LongCite デプロイ

**環境設定**:
`transformers>=4.43.0` を使用することをお勧めします。

私たちは2つのモデルをオープンソース化しました: [LongCite-glm4-9b](https://huggingface.co/THUDM/LongCite-glm4-9b) と [LongCite-llama3.1-8b](https://huggingface.co/THUDM/LongCite-llama3.1-8b)。これらのモデルはそれぞれ [GLM-4-9B](https://huggingface.co/THUDM/glm-4-9b) と [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B) に基づいてトレーニングされており、最大128Kのコンテキストをサポートしています。これらのモデルは論文中の "LongCite-9B" と "LongCite-8B" モデルに対応しています。長文コンテキストに基づくクエリを与えると、これらのモデルは正確な回答と精密な文レベルの引用を生成し、ユーザーが出力情報を検証しやすくします。モデルを試してみてください:
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
以下のコマンドを実行して、独自の LongCite チャットボットをデプロイできます（ビデオでデモンストレーションしたもののように）：
```
CUDA_VISIBLE_DEVICES=0 streamlit run demo.py --server.fileWatcherType none
```
また、[vllm](https://github.com/vllm-project/vllm) を使用してモデルをデプロイすることもできます。これにより、生成が高速化され、マルチコンカレントサーバーがサポートされます。コード例は [vllm_inference.py](https://github.com/THUDM/LongCite/blob/main/vllm_inference.py) にあります。

<a name="パイプライン"></a>
## 🤖️ データ構築パイプライン CoF
![cof](https://github.com/user-attachments/assets/dae25838-3ce0-4a2c-80f7-307c8128e5c4)

私たちは、`CoF/` ディレクトリに CoF（Coarse to Fine）をオープンソース化しました。これは、細粒度の引用を含む高品質な長文コンテキストQAインスタンスを生成するための自動化されたSFTデータ構築パイプラインです。`utils/llm_api.py` にAPIキーを設定し、次の4つのスクリプトを実行して最終データを取得してください：`1_qa_generation.py`、`2_chunk_level_citation.py`、`3_sentence_level_citation.py`、および `4_postprocess_and_filter.py`。

<a name="トレーニング"></a>

## 🖥️ モデルトレーニング

Hugging Face データセット ([🤗 HF Repo](https://huggingface.co/datasets/THUDM/LongCite-45k)) を通じて **LongCite-45k** データセットをダウンロードして保存できます：
```python
dataset = load_dataset('THUDM/LongCite-45k')
for split, split_dataset in dataset.items():
    split_dataset.to_json("train/long.jsonl")
```
これを [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/tree/main/HTML_cleaned_raw_dataset) などの一般的なSFTデータと混合して使用できます。私たちは [Metragon-LM](https://github.com/NVIDIA/Megatron-LM) を使用してモデルをトレーニングします。より軽量な実装を希望する場合は、[LongAlign](https://github.com/THUDM/LongAlign) のコードと環境を採用することができます。これにより、GLM-4-9B および Llama-3.1-8B の最大トレーニングシーケンス長32kトークンをサポートできます。

<a name="評価"></a>
## 📊 評価
私たちは、長文コンテキストQAシナリオにおける引用の質と応答の正確性を測定するための自動ベンチマーク：**LongBench-Cite** を導入しました。これは [LongBench](https://github.com/THUDM/LongBench) および [LongBench-Chat](https://github.com/THUDM/LongAlign) から長文コンテキストQAペアを採用しています。

評価データとコードは `LongBench-Cite/` ディレクトリに提供されています。`pred_sft.py` および `pred_one_shot.py` を実行して、微調整されたモデル（例：LongCite-glm4-9b）および通常のモデル（例：GPT-4o）からの応答を取得します。次に、`eval_cite.py` および `eval_correct.py` を実行して、引用の質と応答の正確性を評価します。`utils/llm_api.py` にOpenAI APIキーを設定することを忘れないでください。私たちはGPT-4oを審査員として採用しています。

以下は **LongBench-Cite** の評価結果です：
![eval_results](https://github.com/user-attachments/assets/1ef68c5f-63f0-4041-8f19-b8694b0d68c2)

<a name="引用"></a>
## 📝 引用

私たちの仕事が役に立つと思われた場合は、LongCiteを引用してください：

```
@article{zhang2024longcite,
  title = {LongCite: Enabling LLMs to Generate Fine-grained Citations in Long-context QA} 
  author={Jiajie Zhang and Yushi Bai and Xin Lv and Wanjun Gu and Danqing Liu and Minhao Zou and Shulin Cao and Lei Hou and Yuxiao Dong and Ling Feng and Juanzi Li},
  journal={arXiv preprint arXiv:2409.02897},
  year={2024}
}
```
