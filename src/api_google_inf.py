import sys
from typing import List, Any
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
import numpy as np

genai.configure()

# --- ヘルパー関数 -----------------------------------------------------------------

def _get_model_name(settings: Any, prefix: str) -> str:
    """
    settingsオブジェクトとプレフィックスに基づき、モデル名を取得する。
    """
    return getattr(settings, f"{prefix}_model_name")

def _get_sampling_params(settings: Any, prefix: str) -> dict:
    """
    settingsオブジェクトとプレフィックスに基づき、Ollama用のサンプリングパラメータを生成する。
    """
    temperature = getattr(settings, f"{prefix}_temperature", 0.7)
    top_p = getattr(settings, f"{prefix}_top_p", 0.9)
    max_tokens = getattr(settings, "max_tokens", 4096)
    
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

def _execute_inference(model_name: str, prompts: List[str], options: dict, is_chat: bool) -> List[str]:
    """
    Ollamaを使用して推論を実行する内部関数。
    """
    print(f"Googleモデル '{model_name}' を使用して{len(prompts)}件の推論を実行中...")
    results = []
    try:
        model = genai.GenerativeModel(model_name)
        for prompt in prompts:
            generation_config = {
                "temperature": options.get("temperature"),
                "top_p": options.get("top_p"),
                "max_output_tokens": options.get("max_tokens"),
            }
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            if is_chat:
                # For chat models, use start_chat
                response = model.start_chat(history=[]).send_message(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
            else:
                # For text generation models
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
            
            results.append(response.text)
        print("推論が完了しました。")
        return results
    except GoogleAPIError as e:
        print(f"Google APIでの推論中にエラーが発生しました: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)

# --- 1. モデルロード/アンロード系関数（ダミー） --------------------------------------------------
# OpenAIではクライアント側での明示的なロード/アンロードは不要なため、ダミー関数を定義します。

def inst_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "Instruct")
    print(f"Googleモデル '{model_name}' を使用します。")
    return model_name

def base_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "base")
    print(f"Googleモデル '{model_name}' を使用します。")
    return model_name

def think_model_load(settings: Any) -> str:
    model_name = _get_model_name(settings, "think")
    print(f"Googleモデル '{model_name}' を使用します。")
    return model_name

def unload_model(model_path_for_log: str):
    print(f"Googleモデル '{model_path_for_log}' のセッションを終了しました。")
    pass

# --- 2. 推論系関数（5種類） ------------------------------------------------------

def inst_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    Instructモデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "Instruct")
    return _execute_inference(model_name, prompts, options, is_chat=True)

def base_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    ベースモデル（生成形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False)

def think_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    長考モデル（チャット形式）を用いてバッチ推論を行う。
    """
    model_name = llm
    options = _get_sampling_params(settings, "think")
    return _execute_inference(model_name, prompts, options, is_chat=True)

def curation_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    キュレーション用の推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.01
    options["top_p"] = 1.0
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False)

def evolution_model_inference(llm: str, prompts: List[str], settings: Any) -> List[str]:
    """
    思考を進化させるための推論を行う。ベースモデルを使用するが、サンプリングパラメータが異なる。
    """
    model_name = llm
    options = _get_sampling_params(settings, "base")
    options["temperature"] = 0.6
    options["top_p"] = 0.9
    options["stop"] = ['<stop>']
    return _execute_inference(model_name, prompts, options, is_chat=False)

# --- 埋め込み関数 -----------------------------------------------------------------

def get_embeddings(sentences: List[str], settings: Any) -> np.ndarray:
    model_name = getattr(settings, 'E5_model_name', 'models/embedding-001')
    print(f"Google Gemini埋め込みモデル '{model_name}' を使用してベクトルを生成します。")
    try:
        # Google Geminiの埋め込みモデルは、テキストのリストを受け取って埋め込みを生成する
        # genai.embed_content を使用
        response = genai.embed_content(
            model=model_name,
            content=sentences,
            task_type="RETRIEVAL_DOCUMENT"
        )
        # 埋め込みは response.embedding にリストとして格納されている
        if response and response['embedding']:
            return np.array(response['embedding'])
        else:
            print(f"Google Geminiからの予期せぬレスポンス形式です: {response}")
            sys.exit(1)
    except GoogleAPIError as e:
        print(f"Google APIでの埋め込み生成中にエラーが発生しました: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        sys.exit(1)
