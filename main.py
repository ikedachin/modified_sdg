#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
from datetime import timedelta
from src.funs import setup_pipeline

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Synthetic Data Generation Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="settings.yaml",
        help="Path to the settings YAML file."
    )
    args = parser.parse_args()

    # パイプライン初期化
    pipe = setup_pipeline(args.config)

    try:
        # 質問生成
        questions = pipe.generate_questions()
        # ベースモデル生成時のみキュレーション
        questions = pipe.curate_questions(questions)
        # 多様性フィルタ
        time.sleep(10)
        questions = pipe.diversity_filter(questions)
        # 質問進化
        time.sleep(10)
        questions = pipe.evolve_questions(questions)
        # 回答生成
        time.sleep(10)
        answers = pipe.generate_answers(questions)
        # print(answers)
        # 回答進化
        time.sleep(10)
        answers = pipe.evolve_answers(answers)
        # 思考生成
        time.sleep(10)
        answers = pipe.generate_thinking(answers)
        # 思考進化
        time.sleep(10)
        # answers = pipe.evolve_thinking(answers)
        # print(answers)
        # 最終キュレーション
        time.sleep(10)
        final_data = pipe.curate_final(answers)
        # print(final_data)
        # 保存
        output_path = pipe.save_dataset(final_data)
        print(f"データセット生成完了: {output_path.resolve()}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    time_formatted = str(timedelta(seconds=int(elapsed_time)))
    print(f"総所要時間: {time_formatted}")

if __name__ == "__main__":
    main()
