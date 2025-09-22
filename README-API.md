# 環境構築ガイド (API)

このガイドは、APIを利用して推論実行するための手順を解説します。

## 設定方法

settings_api.yaml を編集してください

### 1 OpenAI APIを利用する場合

```yaml
# - settings_api.yaml に以下を追加してください

inference_backend: "api_openai"
openai_api_key: "YOUR_OPENAI_API_KEY"
openai_base_url: "https://api.openai.com/v1"
Instruct_model_name: "gpt-4.1-nano"
base_model_name: "gpt-4.1-nano"
think_model_name: "gpt-4o"
E5_model_name: "text-embedding-3-small"
```

### 2 Google APIを利用する場合

```yaml
# - settings_api.yaml に以下を追加してください

inference_backend: "api_google"
google_api_key: "YOUR_GOOGLE_API_KEY"
Instruct_model_name: "gemini-2.5-flash-lite"
base_model_name: "gemini-2.5-flash-lite"
think_model_name: "gemini-2.5-flash-lite"
E5_model_name: "gemini-embedding-001"
```
### 3 OpenAI互換APIを利用する場合

#### 3-1 独自LLMサーバー

ローカルPCやレンタルサーバー上のAPI環境（FastAPI、OpenRouter等）を
利用する場合はこちら

```yaml
# - settings_api.yaml に以下を追加してください

inference_backend: "api_openai_comp"
openai_comp_api_key: "YOUR_OPENAI_COMPLIANT_API_KEY" # ← 不要な場合はコメントアウト
openai_comp_endpoint: "http://localhost:1234/v1"    # ← 独自LLMサーバーのエンドポイントを指定
Instruct_model_name: "google/gemma-3-1b"
base_model_name: "google/gemma-3-1b"
think_model_name: "deepseek-r1-distill-qwen-1.5b"
E5_model_name: "text-embedding-mxbai-embed-large-v1"
```

#### 3-2 Groq（Xじゃない方）

※注意） embedモデルが見当たらず、処理が途中で落ちます。　後日対応します

```yaml
# - settings_api.yaml に以下を追加してください

inference_backend: "api_openai_comp"
openai_comp_api_key: "YOUR_GROQ_API_KEY"
openai_comp_endpoint: "https://api.groq.com/openai/v1"
Instruct_model_name: "google/gemma-3-1b"
base_model_name: "google/gemma-3-1b"
think_model_name: "qwen/qwen3-32b"
# Groqにembedモデルが無い　どうするか？
E5_model_name: "text-embedding-mxbai-embed-large-v1"
```
#### 3-3 SambaNova

※注意） embedモデルは有るが、やはり処理が途中で落ちます。　後日対応します

```yaml
# - settings_api.yaml に以下を追加してください

inference_backend: "api_openai_comp"
openai_comp_api_key: "YOURE_SambaNova_API_KEY"
openai_comp_endpoint: "https://api.sambanova.ai/v1"
Instruct_model_name: "Meta-Llama-3.1-8B-Instruct"
base_model_name: "Meta-Llama-3.1-8B-Instruct"
think_model_name: "Qwen3-32B"
E5_model_name: "E5-Mistral-7B-Instruct"
```

### 4 その他

必要に応じて以下を設定

```yaml

Instruct_temperature: 0.7
Instruct_top_p: 1.0
base_temperature: 0.7
base_top_p: 1.0
think_temperature: 0.7
think_top_p: 1.0

```


## 実行

API用の設定ファイル `settings_api.yaml` を使って実行します。

```bash
python main.py --config settings_api.yaml
```
