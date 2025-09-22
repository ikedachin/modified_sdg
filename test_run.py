"""
quick_test_vllm.py
修正した src/vllm_inf.py が import パス上にある前提
"""

from types import SimpleNamespace
from src.vllm_inf import inst_model_load, inst_model_inference, unload_model
from src.util import load_config

prompts = []

for i in range(32):
    prompts.append("量子力学における「重ね合わせ」と「もつれ」の概念を説明してください。")

settings = load_config('settings.yaml')

# --- モデルロード & 推論 ----------------------------------------------
llm = inst_model_load(settings)

outputs = inst_model_inference(llm, prompts, settings)

print("outputs:", outputs[31])

unload_model(settings.Instruct_model_name)
