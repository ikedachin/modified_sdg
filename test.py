#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import shutil
import math
from pathlib import Path
from itertools import islice
from typing import List, Dict, Any, Tuple

from tqdm import tqdm

# 自作モジュールのインポート
from src import util, vllm_inf, e5

# --- ヘルパー関数 -----------------------------------------------------------------

def batched(iterable, n):
    """
    イテラブルをサイズnのバッチに分割するヘルパー関数。
    例: batched('ABCDEFG', 3) -> ABC DEF G
    """
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

# --- パイプラインの各ステップに対応する関数 -----------------------------------------

def setup_directories(settings: Any) -> Tuple[Path, Path]:
    """出力先ディレクトリを準備する"""
    data_dir = Path(settings.data_folda_path)
    output_dir = Path(settings.output_path)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"一時データフォルダ: {data_dir.resolve()}")
    print(f"最終出力フォルダ: {output_dir.resolve()}")
    return data_dir, output_dir

def load_prompts() -> Dict[str, str]:
    """プロンプトファイルを一括で読み込む"""
    prompt_files = {
        "bare": "./prompts/bare.txt",
        "inst_seed": "./prompts/inst_seed.txt",
        "curation_base": "./prompts/curation_prompt_4_base.txt",
        "evo_question": "./prompts/evo_question_prompt.txt",
        "evo_answer": "./prompts/evo_answer_prompt.txt",
        "evo_thinking": "./prompts/evo_thinking_prompt.txt",
        "curation": "./prompts/curation.txt",
        "thinking_prompt": "./prompts/thinking_prompt.txt",
        "answer_prompt": "./prompts/answer_prompt.txt",

    }
    prompts = {}
    print("\nプロンプトファイルを読み込んでいます...")
    for name, path in prompt_files.items():
        try:
            prompts[name] = util.read_text_file(path)
            print(f"  - {name}: 読み込み成功")
        except FileNotFoundError:
            raise FileNotFoundError(f"プロンプトファイルが見つかりません: {path}")
    return prompts

def generate_questions(settings: Any, prompts: Dict[str, str], data_dir: Path) -> List[Dict]:
    """手順1: 質問データを生成する"""
    print("\n--- ステップ1: 質問データの生成を開始 ---")
    
    # 生成方式に応じてモデルをロード
    if settings.Seed_generation_method == 'base':
        print("ベースモデルを使用して質問を生成します。")
        model = vllm_inf.base_model_load(settings)
        inference_func = vllm_inf.base_model_inference
        prompt_template = prompts["bare"]
    elif settings.Seed_generation_method == 'inst':
        print("Instructモデルを使用して質問を生成します。")
        model = vllm_inf.inst_model_load(settings)
        inference_func = vllm_inf.inst_model_inference
        prompt_template = prompts["inst_seed"]  # 指示モデルでも同じプロンプトを使用
    else:
        raise ValueError("settings.yaml の Seed_generation_method は 'base' または 'inst' である必要があります。")

    questions_file = data_dir / "generated_questions.jsonl"
    # 毎回クリーンにしたい場合は既存ファイルを削除
    if questions_file.exists():
        questions_file.unlink()
        print(f"既存ファイル {questions_file.name} を削除して新規作成します。")

    generated_count = 0
    pbar = tqdm(total=settings.Number_of_questions_generated, desc="質問生成中")
    
    while generated_count < settings.Number_of_questions_generated:
        batch_size = settings.batch_size
        
        # ▼▼▼ 修正箇所 ▼▼▼
        # バッチ内の各プロンプトに適用する、異なるノイズを事前に生成します。
        num_nouns_per_prompt = 5
        total_nouns_to_generate = batch_size * num_nouns_per_prompt
        
        try:
            noise_nouns = util.random_japanese_nouns(total_nouns_to_generate)
        except ValueError as e:
            print(f"警告: ノイズ生成に必要な数のユニークな名詞を取得できませんでした。: {e}")
            print("このバッチの処理をスキップします。バッチサイズを減らすか、名詞のプールサイズを増やしてください。")
            continue

        batch_prompts = []
        for i in range(batch_size):
            start_index = i * num_nouns_per_prompt
            end_index = start_index + num_nouns_per_prompt
            nouns_for_prompt = noise_nouns[start_index:end_index]
            noise = ' '.join(nouns_for_prompt)
            prompt = f"ノイズ: {noise}\n\n{prompt_template}"
            batch_prompts.append(prompt)
        # ▲▲▲ 修正箇所 ▲▲▲

        results = inference_func(model, batch_prompts, settings)
        
        batch_data = []
        for res in results:
            if not res:
                continue

            if settings.Seed_generation_method == 'base':
                for line in res.strip().split('\n'):
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and "Question" in data and data["Question"]:
                            batch_data.append(data)
                    except json.JSONDecodeError:
                        continue
            else:
                batch_data.append({"Question": res.strip(), "Answer": ""})
        
        if batch_data:
            util.save_jsonl(batch_data, str(questions_file), mode='a')
            generated_count += len(batch_data)
            pbar.update(len(batch_data))

    pbar.close()
    
    model_name_for_log = settings.base_model_name if settings.Seed_generation_method == 'base' else settings.Instruct_model_name
    del model
    vllm_inf.unload_model(model_name_for_log)

    # ------------------------------------------------------------------- #
    # ▼▼▼ 修正箇所: ファイル読み込みと重複削除 ▼▼▼
    # ------------------------------------------------------------------- #
    all_questions = util.load_jsonl(str(questions_file))
    print(f"合計 {len(all_questions)} 件の質問を生成しました。")

    print("生成された質問の重複を削除しています...")
    seen_questions = set()
    unique_questions = []
    for item in all_questions:
        question_text = item.get("Question")
        if question_text and question_text.strip():
            normalized_question = question_text.strip()
            if normalized_question not in seen_questions:
                unique_questions.append(item)
                seen_questions.add(normalized_question)

    num_duplicates = len(all_questions) - len(unique_questions)
    if num_duplicates > 0:
        print(f"{num_duplicates} 件の重複または空の質問を削除しました。")
    print(f"重複削除後の質問数: {len(unique_questions)} 件")
    
    return unique_questions
    # ▲▲▲ 修正箇所ここまで ▲▲▲

### --- PATCH START -------------------------------------------------------------
def curate_base_questions(data: List[Dict], settings: Any, prompts: Dict[str, str]) -> List[Dict]:
    """
    ベースモデルで生成した質問データをキュレーションする。
    旧実装ではチャンクごとに逐次推論しており GPU 利用効率が悪かった。
    本実装では「チャンク化 → チャンク列をさらにバッチ化」という二段階の
    バッチ推論を行い、必ず `settings.batch_size` 件ずつモデルに投入する。
    """
    print("\n--- ベースモデル生成データのキュレーションを実行 ---")
    if not data:
        print("データが空のためスキップします。")
        return []

    model = vllm_inf.inst_model_load(settings)
    curation_prompt = prompts["curation_base"]

    # 1. 質問を小さなチャンク（例: 4件）にまとめて 1 つのプロンプトを作る
    chunk_size = 4
    data_chunks: List[List[Dict]] = list(batched(data, chunk_size))

    # 2. (chunk, prompt) のタプルを作成
    chunk_prompt_pairs: List[Tuple[List[Dict], str]] = [
        (
            chunk,
            f"{curation_prompt}\n\n{ ' + '.join(d.get('Question', '') for d in chunk) }"
        )
        for chunk in data_chunks
    ]

    curated_data: List[Dict] = []
    total_steps = math.ceil(len(chunk_prompt_pairs) / settings.batch_size)
    pbar = tqdm(total=total_steps, desc="ベースデータキュレーション中")

    # 3. チャンク列を settings.batch_size 件ずつまとめて推論
    for pair_batch in batched(chunk_prompt_pairs, settings.batch_size):
        batch_chunks, batch_prompts = zip(*pair_batch)
        results = vllm_inf.inst_model_inference(model, list(batch_prompts), settings)

        for res, chunk in zip(results, batch_chunks):
            if "yes" in res.lower():
                curated_data.extend(chunk)
        pbar.update(1)

    pbar.close()

    del model
    vllm_inf.unload_model(settings.Instruct_model_name)
    
    print(f"キュレーションの結果、{len(data)}件から{len(curated_data)}件のデータが残りました。")
    return curated_data
### --- PATCH END ---------------------------------------------------------------

def filter_by_diversity(data: List[Dict], settings: Any) -> List[Dict]:
    """手順2: 質問データの多様性フィルタリング"""
    print("\n--- ステップ2: 質問データの多様性フィルタリングを開始 ---")
    if not data:
        print("データが空のためスキップします。")
        return []

    retention_rate = settings.Data_retention_rate_after_diversity_cut
    cut_rate = 100.0 - retention_rate

    filtered_data = e5.reduce_data_by_diversity(
        data=data,
        data_format='dict',
        cut_rate=cut_rate,
        data_key='Question',
        model_path=settings.E5_path
    )
    return filtered_data

def evolve_pipeline(original_data: List[Dict], settings: Any, prompts: Dict[str, str], evolution_type: str) -> List[Dict]:
    """手順3 & 5: 質問または回答の進化パイプライン"""
    
    if evolution_type == "question":
        enable_flag = settings.Prompt_evolution
        times = settings.Prompt_evolution_times
        staged_flag = settings.Number_of_stages_of_prompt_evolution
        evo_prompt = prompts["evo_question"]
        data_key = "Question"
        print_prefix = "質問"
    elif evolution_type == "answer":
        enable_flag = settings.Answer_evolution
        times = settings.Answer_evolution_times
        staged_flag = settings.Number_of_stages_of_answer_evolution
        evo_prompt = prompts["evo_answer"]
        data_key = "Answer"
        print_prefix = "回答"
    else:
        raise ValueError("evolution_typeは 'question' または 'answer' である必要があります。")

    if not enable_flag or times == 0:
        print(f"\n--- {print_prefix}の進化はスキップされます ---")
        return original_data

    print(f"\n--- ステップ3/5: {print_prefix}の進化を開始 (進化回数: {times}回) ---")
    if not original_data:
        print("データが空のためスキップします。")
        return []

    model = vllm_inf.inst_model_load(settings)
    
    evolved_data_stages = {0: original_data}
    data_to_evolve = [d.copy() for d in original_data]

    for stage_idx in range(1, times + 1):
        print(f"\n進化ステージ {stage_idx}/{times} を実行中...")
        current_stage_data = []
        pbar = tqdm(total=len(data_to_evolve), desc=f"{print_prefix}進化 {stage_idx}回目")
        for batch in batched(data_to_evolve, settings.batch_size):
            batch_prompts = [f"{evo_prompt}\n\n{d[data_key]}" for d in batch]
            
            evolved_texts = vllm_inf.inst_model_inference(model, batch_prompts, settings)
            
            for i, d in enumerate(batch):
                new_d = d.copy()
                new_d[data_key] = evolved_texts[i]
                current_stage_data.append(new_d)
            pbar.update(len(batch))
        
        pbar.close()
        evolved_data_stages[stage_idx] = current_stage_data
        data_to_evolve = current_stage_data

    if staged_flag and times > 0:
        print("\n段階的進化データを構築しています...")
        final_evolved_data = []
        num_stages = times + 1  # 元データも含む
        base_count_per_stage = len(original_data) // num_stages
        
        for stage_idx in range(num_stages):
            stage_data = evolved_data_stages.get(stage_idx, [])
            count = base_count_per_stage if stage_idx < times else len(original_data) - len(final_evolved_data)
            final_evolved_data.extend(random.sample(stage_data, min(count, len(stage_data))))

        print(f"段階的進化の結果、{len(final_evolved_data)}件のデータが構築されました。")
        result_data = final_evolved_data
    else:
        result_data = evolved_data_stages[times]

    del model
    vllm_inf.unload_model(settings.Instruct_model_name)
    return result_data

def generate_answers(data: List[Dict], settings: Any) -> List[Dict]:
    """手順4: 回答データを生成する"""
    print("\n--- ステップ4: 回答データの生成を開始 ---")
    if not data:
        print("データが空のためスキップします。")
        return []

    if settings.Using_think_models_for_answer:
        print("長考モデルを使用して回答を生成します。")
        model = vllm_inf.think_model_load(settings)
        inference_func = vllm_inf.think_model_inference
        model_name_for_log = settings.think_model_name
    else:
        print("Instructモデルを使用して回答を生成します。")
        model = vllm_inf.inst_model_load(settings)
        inference_func = vllm_inf.inst_model_inference
        model_name_for_log = settings.Instruct_model_name

    instruction = "以下の質問に日本語で答えてください。"
    
    answered_data = []
    pbar = tqdm(total=len(data), desc="回答生成中")
    for batch in batched(data, settings.batch_size):
        batch_prompts = [f"{instruction}\n\n{d['Question']}" for d in batch]
        
        results = inference_func(model, batch_prompts, settings)
        
        for i, res in enumerate(results):
            d = batch[i].copy()
            if settings.Using_think_models_for_answer:
                think_part, answer_part = util.separate_think_and_answer(res)
                d["think"] = think_part
                d["Answer"] = answer_part
            else:
                d["Answer"] = res
            answered_data.append(d)
        pbar.update(len(batch))

    pbar.close()
    
    del model
    vllm_inf.unload_model(model_name_for_log)
    print(f"{len(answered_data)}件の回答を生成しました。")
    return answered_data

def curate_final_data(data: List[Dict], settings: Any, prompts: Dict[str, str]) -> List[Dict]:
    """最終的なデータセットをキュレーションする"""
    if not settings.Data_curation:
        print("\n--- 最終データのキュレーションはスキップされます ---")
        return data

    print("\n--- 最終データのキュレーションを開始 ---")
    if not data:
        print("データが空のためスキップします。")
        return []

    model = vllm_inf.inst_model_load(settings)
    curation_prompt = prompts["curation"]
    
    curated_data = []
    pbar = tqdm(total=len(data), desc="最終データキュレーション中")
    for batch in batched(data, settings.batch_size):
        batch_prompts = []
        for d in batch:
            prompt = f'{curation_prompt}\n\n質問: {d["Question"]}\n回答: {d["Answer"]}'
            batch_prompts.append(prompt)
        
        results = vllm_inf.inst_model_inference(model, batch_prompts, settings)
        
        for i, res in enumerate(results):
            if "yes" in res.lower():
                curated_data.append(batch[i])
        pbar.update(len(batch))

    pbar.close()
    
    del model
    vllm_inf.unload_model(settings.Instruct_model_name)
    
    print(f"キュレーションの結果、{len(data)}件から{len(curated_data)}件のデータが残りました。")
    return curated_data

def format_and_save_data(data: List[Dict], settings: Any, output_dir: Path):
    """最終的なデータを指定のJSONLフォーマットに変換して保存する"""
    print("\n--- 最終的なデータフォーマットへの変換と保存 ---")
    if not data:
        print("保存するデータがありません。")
        return

    final_dataset = []
    for item in data:
        formatted_item = {
            "instruction": "以下の質問に日本語で答えてください。",  # 固定のシステムプロンプト
            "input": item.get("Question", ""),
            "output": item.get("Answer", "")
        }
        final_dataset.append(formatted_item)

    output_file = output_dir / "final_dataset.jsonl"
    util.save_jsonl(final_dataset, str(output_file), mode='w')
    print(f"最終データセット {len(final_dataset)}件 を {output_file.resolve()} に保存しました。")

def cleanup(settings: Any, data_dir: Path):
    """一時ファイルをクリーンアップする"""
    if not settings.Save_temporary_data:
        print(f"\n一時データフォルダ {data_dir.resolve()} を削除します。")
        try:
            shutil.rmtree(data_dir)
            print("クリーンアップが完了しました。")
        except OSError as e:
            print(f"エラー: {data_dir} の削除に失敗しました - {e.strerror}")

# --- メイン実行部 ---------------------------------------------------------------

def main():
    """データ生成パイプラインのメイン関数"""
    try:
        # 1. 設定とプロンプトの読み込み
        settings = util.load_config("settings.yaml")
        prompts = load_prompts()
        
        # 2. ディレクトリの準備
        data_dir, output_dir = setup_directories(settings)

        # 3. パイプラインの実行
        generated_data = generate_questions(settings, prompts, data_dir)
        
        if settings.Seed_generation_method == 'base':
            generated_data = curate_base_questions(generated_data, settings, prompts)

        filtered_data = filter_by_diversity(generated_data, settings)
        evolved_questions_data = evolve_pipeline(filtered_data, settings, prompts, "question")
        answered_data = generate_answers(evolved_questions_data, settings)
        evolved_answers_data = evolve_pipeline(answered_data, settings, prompts, "answer")
        final_data = curate_final_data(evolved_answers_data, settings, prompts)
        format_and_save_data(final_data, settings, output_dir)
        cleanup(settings, data_dir)
        
        print("\nデータ生成パイプラインが正常に完了しました。")

    except (util.ConfigLoadError, FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\nエラーが発生したため処理を中断しました: {e}", file=os.sys.stderr)
        vllm_inf.destroy_model_parallel()

if __name__ == "__main__":
    main()
