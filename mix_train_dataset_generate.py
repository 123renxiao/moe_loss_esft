"""
混合训练数据集生成器
========================
本脚本从多个不同领域的数据集中混合采样，生成用于多任务学习的训练数据。
支持7种数据集，每种数据集对应一个专家标签（_tag字段）。

作者：Li Wentian
日期：2025年12月
"""

import json
import random
import os
import argparse
import csv
import sys
from datasets import load_from_disk

# 增加CSV字段大小限制，以支持包含长文本的数据
csv.field_size_limit(sys.maxsize)

# ================= 配置路径 =================
BASE_DIR = "/data/storage2/liwentian"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# 1. ArkitScenes
arkitscenes_path = os.path.join(BASE_DIR, "llm_pipe_gen/arkitscenes-llm/arkitscenes_train.json")

# 2. MATH-500
math500_path = os.path.join(DATASET_DIR, "local_math500")

# 3. GSM8K
gsm8k_path = os.path.join(DATASET_DIR, "GSM8K")

# 4. MetaMathQA
metamath_path = os.path.join(DATASET_DIR, "MetaMathQA")

# 5. Intent Recognition
intent_path = os.path.join(DATASET_DIR, "Intent_recognition.json")

# 6. Law
law_path = os.path.join(DATASET_DIR, "law.json")

# 7. Text Summary
text_summary_path = os.path.join(DATASET_DIR, "text_summary.csv")

# 输出路径
output_path = os.path.join(BASE_DIR, "llm_pipe_gen/arkitscenes-llm/mixed_train_dataset_all_7.json")

# ================= 辅助函数 =================

def format_sharegpt(input_text, output_text, tag_id):
    """
    将输入输出文本转换为ShareGPT对话格式
    
    参数:
        input_text: 用户输入（问题/任务描述）
        output_text: 模型输出（答案/结果）
        tag_id: 专家标签ID，用于区分不同领域的数据
    
    返回:
        字典，包含conversations列表和_tag字段
    """
    return {
        "conversations": [
            {
                "from": "human",
                "value": str(input_text)
            },
            {
                "from": "gpt",
                "value": str(output_text)
            }
        ],
        "_tag": tag_id  # 专家标签，MoE模型可以利用此标签进行专家路由
    }

def sample_data(data_list, num_samples):
    """
    从数据列表中随机采样指定数量的样本
    
    参数:
        data_list: 原始数据列表
        num_samples: 需要采样的数量，0或负数表示使用全部数据
    
    返回:
        采样后的数据列表
    """
    if num_samples > 0 and len(data_list) > num_samples:
        return random.sample(data_list, num_samples)
    return data_list

# ================= 加载函数 =================

def load_hf_dataset(path, split_name, num_samples, tag_id, input_col, output_col):
    """
    从HuggingFace格式的数据集中加载数据
    
    参数:
        path: 数据集路径（使用load_from_disk加载）
        split_name: 数据集分割名称（如'train', 'test'）
        num_samples: 采样数量
        tag_id: 专家标签ID
        input_col: 输入列名
        output_col: 输出列名
    
    返回:
        ShareGPT格式的数据列表
    """
    print(f"[{tag_id}] 正在加载 HF 数据: {path} ...")
    formatted_data = []
    try:
        # 加载数据集字典
        dataset_dict = load_from_disk(path)
        
        # 检查split是否存在，不存在则使用第一个可用的split
        if split_name not in dataset_dict:
            available = list(dataset_dict.keys())
            if not available:
                print(f"  ❌ 错误: 数据集为空")
                return []
            split_name = available[0]
            print(f"  ⚠️ Split 不存在，使用 '{split_name}'")
        
        dataset = dataset_dict[split_name]
        total_len = len(dataset)
        
        # 采样索引，避免加载全部数据到内存
        if num_samples > 0 and total_len > num_samples:
            indices = random.sample(range(total_len), num_samples)
            selected_items = [dataset[i] for i in indices]
        else:
            selected_items = list(dataset)

        # 转换为ShareGPT格式
        for item in selected_items:
            formatted_data.append(format_sharegpt(item[input_col], item[output_col], tag_id))
            
        print(f"  ✅ 完成，抽取 {len(formatted_data)} 条")
        return formatted_data
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return []

def load_json_dict(path, num_samples, tag_id, input_col, output_col):
    print(f"[{tag_id}] 正在加载 JSON Dict: {path} ...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 转换为列表
        data_list = list(data.values())
        selected_items = sample_data(data_list, num_samples)
        
        formatted_data = []
        for item in selected_items:
            formatted_data.append(format_sharegpt(item[input_col], item[output_col], tag_id))
            
        print(f"  ✅ 完成，抽取 {len(formatted_data)} 条")
        return formatted_data
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return []

def load_json_lines(path, num_samples, tag_id, input_col, output_col):
    print(f"[{tag_id}] 正在加载 JSON Lines: {path} ...")
    data_list = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_list.append(json.loads(line))
        
        selected_items = sample_data(data_list, num_samples)
        
        formatted_data = []
        for item in selected_items:
            # 特殊处理 Law 数据集的 output (list -> string)
            out_val = item[output_col]
            if isinstance(out_val, list):
                out_val = "\n".join(out_val)
            
            formatted_data.append(format_sharegpt(item[input_col], out_val, tag_id))
            
        print(f"  ✅ 完成，抽取 {len(formatted_data)} 条")
        return formatted_data
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return []

def load_csv(path, num_samples, tag_id, input_col, output_col, delimiter='|'):
    print(f"[{tag_id}] 正在加载 CSV: {path} ...")
    data_list = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                data_list.append(row)
        
        selected_items = sample_data(data_list, num_samples)
        
        formatted_data = []
        for item in selected_items:
            formatted_data.append(format_sharegpt(item[input_col], item[output_col], tag_id))
            
        print(f"  ✅ 完成，抽取 {len(formatted_data)} 条")
        return formatted_data
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return []

def load_arkitscenes(path, num_samples, tag_id):
    print(f"[{tag_id}] 正在加载 ArkitScenes: {path} ...")
    try:
        if not os.path.exists(path):
            print(f"  ❌ 文件不存在")
            return []
            
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        selected_items = sample_data(data, num_samples)
        
        # ArkitScenes 已经是 ShareGPT 格式，只需要加 tag
        for item in selected_items:
            item['_tag'] = tag_id
            
        print(f"  ✅ 完成，抽取 {len(selected_items)} 条")
        return selected_items
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return []

# ================= 主逻辑 =================

def process_datasets(args):
    combined_data = []
    
    # 1. ArkitScenes (Tag 0)
    combined_data.extend(load_arkitscenes(arkitscenes_path, args.num_samples, 0))
    
    # 2. MATH-500 (Tag 1)
    combined_data.extend(load_hf_dataset(math500_path, 'test', args.num_samples, 1, 'problem', 'solution'))
    
    # 3. GSM8K (Tag 2)
    combined_data.extend(load_hf_dataset(gsm8k_path, 'train', args.num_samples, 2, 'question', 'answer'))
    
    # 4. MetaMathQA (Tag 3)
    combined_data.extend(load_hf_dataset(metamath_path, 'train', args.num_samples, 3, 'query', 'response'))
    
    # 5. Intent Recognition (Tag 4)
    combined_data.extend(load_json_dict(intent_path, args.num_samples, 4, 'text', 'intent'))
    
    # 6. Law (Tag 5)
    combined_data.extend(load_json_lines(law_path, args.num_samples, 5, 'text', 'case_result'))
    
    # 7. Text Summary (Tag 6)
    combined_data.extend(load_csv(text_summary_path, args.num_samples, 6, 'content', 'abstract', delimiter='|'))

    # --- 保存 ---
    print("\n正在打乱数据顺序...")
    random.shuffle(combined_data)
    
    print(f"正在保存混合数据集到: {output_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
    print(f"🎉 成功！混合数据集已生成。")
    print(f"总数据量: {len(combined_data)}")
    
    # 统计各 Tag 数量
    from collections import Counter
    tags = Counter([item.get('_tag') for item in combined_data])
    tag_names = {
        0: "ArkitScenes",
        1: "MATH-500",
        2: "GSM8K",
        3: "MetaMathQA",
        4: "Intent",
        5: "Law",
        6: "Summary"
    }
    for tag_id, count in sorted(tags.items()):
        print(f"  - Tag {tag_id} ({tag_names.get(tag_id, 'Unknown')}): {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成全量混合训练数据集")
    parser.add_argument("--num_samples", type=int, default=500, help="每个数据集抽取的样本数量 (默认: 500)")
    
    args = parser.parse_args()
    
    process_datasets(args)
