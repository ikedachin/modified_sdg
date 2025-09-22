import gc
import torch
import time
from vllm import LLM, SamplingParams
from typing import List, Any
from vllm.distributed import destroy_model_parallel
import numpy as np
from sentence_transformers import SentenceTransformer
import sys

# --- ヘルパー関数 -----------------------------------------------------------------

def _load_model_from_settings(settings: Any, prefix: str) -> LLM:
    """
    settingsオブジェクトとプレフィックスに基づき、vLLMモデルをロードする内部関数。
    """
    model_name = getattr(settings, f"{prefix}_model_name")
    quantization = getattr(settings, f"{prefix}_model_quantization", None)
    gpu_memory_utilization = getattr(settings, f"{prefix}_gpu_memory_utilization", 0.9)
    dtype = getattr(settings, f"{prefix}_dtype", "auto")
    
    if quantization and str(quantization).lower() == 'null':
        quantization = None

    seed = getattr(settings, "seed", int(time.time()))

    print(f"モデル '{model_name}' をロードしています...")
    print(f"  - quantization: {quantization}")
    print(f"  - tensor_parallel_size: {settings.tensor_parallel_size}")
    print(f"  - seed: {seed}")

    llm = LLM(
        model=model_name,
        quantization=quantization,
        tensor_parallel_size=settings.tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=settings.trust_remote_code,
        max_model_len=settings.max_tokens,
        seed=seed
    )
    print("モデルのロードが完了しました。")
    return llm

def _execute_inference_with_error_handling(
    llm: LLM, 
    prompts: List[str], 
    sampling_params: SamplingParams
) -> List[str]:
    """
    例外処理を強化した推論実行の内部関数。
    """
    print(f"{len(prompts)}件のバッチ推論を実行中...")
    try:
        outputs = llm.generate(prompts, sampling_params)
        results = [output.outputs[0].text for output in outputs]
        print("バッチ推論が完了しました。")
        return results
    except Exception as e:
        print(f"推論中にエラーが発生しました: {e}")
        return [""] * len(prompts)
    finally:
        gc.collect()
        torch.cuda.empty_cache()

def unload_model(model_path_for_log: str):
    """
    モデルオブジェクトを削除した後に、GPUメモリを解放する。
    """
    destroy_model_parallel()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"モデル '{model_path_for_log}' に関連するリソースを解放し、GPUキャッシュをクリアしました。")


# --- 1. モデルロード系関数（3種類） --------------------------------------------------

def inst_model_load(settings: Any) -> LLM:
    """
    settings.yamlから 'Instruct_' プレフィックスの設定を読み込み、Instructモデルをロードする。
    """
    return _load_model_from_settings(settings, "Instruct")

def base_model_load(settings: Any) -> LLM:
    """
    settings.yamlから 'base_' プレフィックスの設定を読み込み、ベースモデルをロードする。
    """
    return _load_model_from_settings(settings, "base")

def think_model_load(settings: Any) -> LLM:
    """
    settings.yamlから 'think_' プレフィックスの設定を読み込み、長考モデルをロードする。
    """
    return _load_model_from_settings(settings, "think")


# --- 2. 推論系関数（5種類） ------------------------------------------------------

def inst_model_inference(llm: LLM, prompts: List[str], settings: Any) -> List[str]:
    """
    Instructモデルを用いてバッチ推論を行う。プロンプトはChatML形式に変換される。
    """
    seed = getattr(settings, "seed", int(time.time()))
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=settings.Instruct_temperature,
        top_p=settings.Instruct_top_p,
        max_tokens=settings.max_tokens,
        seed=seed,
        stop=[tokenizer.eos_token]
    )
    
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        prompt_str = tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(prompt_str)
        
    return _execute_inference_with_error_handling(llm, formatted_prompts, sampling_params)

def base_model_inference(llm: LLM, prompts: List[str], settings: Any) -> List[str]:
    """
    ベースモデルを用いてバッチ推論を行う。
    """
    sampling_params = SamplingParams(
        temperature=settings.base_temperature,
        top_p=settings.base_top_p,
        max_tokens=settings.max_tokens,
        stop=['<stop>'],
    )
    return _execute_inference_with_error_handling(llm, prompts, sampling_params)

def think_model_inference(llm: LLM, prompts: List[str], settings: Any) -> List[str]:
    """
    長考モデルを用いてバッチ推論を行う。出力には<think>タグが含まれる。
    """
    seed = getattr(settings, "seed", int(time.time()))
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=settings.think_temperature,
        top_p=settings.think_top_p,
        max_tokens=settings.max_tokens,
        seed=seed,
        stop=[tokenizer.eos_token]
    )
    
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        prompt_str = tokenizer.apply_chat_template(
            conversation=messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(prompt_str)
    return _execute_inference_with_error_handling(llm, formatted_prompts, sampling_params)

# --- 埋め込み関数 -----------------------------------------------------------------

def get_embeddings(sentences: List[str], settings: Any) -> np.ndarray:
    model_path = settings.E5_path
    print(f"モデル '{model_path}' を読み込んでいます...")
    try:
        model = SentenceTransformer(model_path)
        return model.encode(['passage: ' + s for s in sentences], show_progress_bar=False)
    except Exception as e:
        print(f"モデル '{model_path}' の読み込みに失敗しました。パスが正しいか確認してください。エラー: {e}")
        sys.exit(1)
