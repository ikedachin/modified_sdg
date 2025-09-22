import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Union, Optional, Any
# import sys

def reduce_data_by_diversity(
    data: Union[List[str], List[Dict]],
    data_format: str,
    cut_rate: float,
    settings: Any,
    data_key: Optional[str] = None,
) -> Union[List[str], List[Dict]]:
    """
    データの多様性に基づいて、類似度の高いデータから指定された割合を削減する関数。
    推論バックエンドの設定に応じて、OllamaまたはローカルのSentenceTransformerを使い分ける。
    """
    # --- 1. 入力値の検証 ---
    if not (0 <= cut_rate <= 100):
        raise ValueError("カット率（cut_rate）は0から100の間で指定してください。")
    if data_format not in ['list', 'dict']:
        raise ValueError("データ形式（data_format）は 'list' または 'dict' を指定してください。")
    if data_format == 'dict' and not data_key:
        raise ValueError("データ形式が 'dict' の場合、データキー（data_key）の指定は必須です。")
    if not data or len(data) < 2:
        return data

    # --- 2. 処理対象の文章リストを抽出 ---
    if data_format == 'list':
        sentences = data
    else: # 'dict'
        try:
            sentences = [item[data_key] for item in data]
        except KeyError:
            raise KeyError(f"指定されたキー '{data_key}' が一部の辞書に存在しません。")
        except TypeError:
            raise TypeError("データ形式が 'dict' の場合、入力データは辞書のリストである必要があります。")

    # --- 3. 削減するデータ数を計算 ---
    num_to_cut = int(len(sentences) * (cut_rate / 100))
    if num_to_cut == 0:
        return data
    if num_to_cut >= len(sentences):
        return []

    # --- 4. バックエンドに応じてエンコード処理を分岐 ---
    backend = getattr(settings, 'inference_backend', 'vllm')
    embeddings = None

    # --- 推論バックエンドの動的インポート ---
    if backend == "ollama":
        from src import ollama_inf as emb_module
        print("推論バックエンドとして Ollama を使用します。")
    elif backend == "api_openai":
        from . import api_openai_inf as emb_module
        print("推論バックエンドとして Google API を使用します。")
    elif backend == "api_google":
        from . import api_google_inf as emb_module
        print("推論バックエンドとして Google API を使用します。")
    elif backend == "api_openai_comp":
        from . import api_openai_comp_inf as emb_module
        print("推論バックエンドとして OpenAI互換 API を使用します。")
    elif backend == "vllm":
        from src import vllm_inf as emb_module
        print("推論バックエンドとして vLLM を使用します。")
    else:
        raise ImportError(f"無効な推論バックエンドが指定されました: {backend}")
    emb = emb_module
    embeddings = emb.get_embeddings(sentences, settings)
    # ---

    # --- 5. 類似度に基づいて削減対象を選定 ---
    if embeddings is not None:
        print("類似度を計算し、削減対象を選定中...")
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, -1)
        redundancy_scores = np.max(similarity_matrix, axis=1)
        indices_to_cut = np.argsort(-redundancy_scores)[:num_to_cut]
        indices_to_cut_set = set(indices_to_cut)

        # --- 6. 削減後のデータを生成 ---
        reduced_data = [item for i, item in enumerate(data) if i not in indices_to_cut_set]
        
        print(f"処理が完了しました。{len(data)}件のデータから{len(indices_to_cut)}件を削減し、{len(reduced_data)}件になりました。")
        return reduced_data
    else:
        # エンべディングが生成されなかった場合は元のデータを返す
        print("警告: エンべディングが生成されなかったため、多様性フィルタリングをスキップしました。")
        return data

