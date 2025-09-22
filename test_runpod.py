#!/usr/bin/env python3
import os
import sys
import json
import requests

# ==== 設定 ====
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://q1dpsx0l6ec0hq-8000.proxy.runpod.net/v1")  # 例: https://<id>-8000.proxy.runpod.net/v1
API_KEY  = os.environ.get("OPENAI_API_KEY", "sk-xxxxx")  # サーバーがAPIキー不要なら値は無視される
MODEL    = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b") # サーバー側でロード済みのモデル名

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def chat_once(user_message: str) -> str:
    """非ストリーミングで1レスポンスを取得"""
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
    }
    try:
        resp = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        print(f"[HTTPError] {e} :: {getattr(e.response, 'text', None)}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        raise

def chat_stream(user_message: str):
    """サーバー送信イベント（SSE）でトークンを逐次受信"""
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": True,  # ストリーミング有効化
    }
    with requests.post(url, headers=HEADERS, data=json.dumps(payload), stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                chunk = line[len("data: "):]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0]["delta"].get("content", "")
                    if delta:
                        print(delta, end="", flush=True)
                except Exception:
                    # SSE以外の行はそのまま表示（必要に応じて無視）
                    pass
        print()

if __name__ == "__main__":
    # 使い方: python client.py "こんにちは！"
    question = sys.argv[1] if len(sys.argv) > 1 else "何か面白い話をしてください。日本語で500文字程度で出力してください。"
    print("=== Non-Streaming ===")
    print(chat_once(question))
    print("\n=== Streaming ===")
    chat_stream(question)