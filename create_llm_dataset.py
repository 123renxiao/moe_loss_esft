"""
LLM训练数据集生成器（用于3D场景布局和管道路径规划）
================================================================
本脚本将3D场景布局数据转换为LLM训练格式，用于训练模型进行：
1. 障碍物和端点检测
2. 最短路径规划
3. 避障路径生成

数据流程:
    原始布局文件 -> Layout解析 -> 归一化/离散化 -> 
    构建提示词 -> 生成训练样本 -> 保存JSON

作者：基于SpatialLM改进
日期：2025年
"""

import os
import argparse
import json
from glob import glob
import traceback

import pandas as pd
from tqdm import tqdm

# 确保引用的是修改后支持 Obstacle/Terminal 的 Layout 类
from spatiallm.layout.layout import Layout

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to the input dataset directory",
    )
    parser.add_argument(
        "-s",
        "--split_csv",
        type=str,
        required=True,
        help="Path to the split csv file",
    )
    parser.add_argument(
        "--code_template_file",
        type=str,
        default="code_template.txt",
    )
    parser.add_argument(
        "-n",
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=640,
        help="Number of bins for discretization",
    )
    args = parser.parse_args()

    layout_dir = os.path.join(args.dataset_dir, "layout")

    # 获取所有 layout 文件
    layout_files = glob(os.path.join(layout_dir, "*.txt"))
    
    # 从文件名提取 scene_id
    scene_ids = [
        os.path.basename(layout_file).split(".")[0] for layout_file in layout_files
    ]

    # 读取 split csv 并过滤
    df = pd.read_csv(args.split_csv, dtype=str)
    df.set_index("id", inplace=True)
    
    # 取交集，确保只处理在 split csv 中的场景
    scene_ids = [sid for sid in scene_ids if sid in df.index]

    print(f"Creating dataset with {len(scene_ids)} scenes...")

    with open(args.code_template_file, "r") as f:
        code_template = f.read()

    dataset = {
        "train": [],
        "val": [],
    }
    
    for si, scene_id in enumerate(tqdm(scene_ids)):
        try:
            # 从CSV中获取该场景的split标签（train或val）
            split = df.loc[scene_id, "split"]

            # 1. 加载布局文件内容
            with open(os.path.join(layout_dir, f"{scene_id}.txt"), "r") as f:
                layout_content = f.read()

            # 2. 解析布局为对象（包含obstacles, terminals等）
            layout = Layout(layout_content)
            
            # 3. 归一化并离散化
            # 将真实坐标归一化到[0,1]，再离散化到[0, num_bins-1]的整数
            # 这样模型可以用token表示坐标，便于序列生成
            layout.normalize_and_discretize(args.num_bins)
            
            # 4. 提取障碍物和端点信息用于构建提示词
            # 此时layout中的实体坐标已经是归一化后的整数
            obstacles = layout.obstacles
            terminals = layout.terminals
            
            # 将障碍物和端点转换为语言描述
            obstacles_str = "\n".join([o.to_language_string() for o in obstacles])
            terminals_str = "\n".join([t.to_language_string() for t in terminals])
            
            # 5. 构建任务提示词
            # 如果有多个端点，生成连接首尾端点的提示
            if len(terminals) >= 2:
                conn_prompt = f"generate a route connecting {terminals[0].entity_label}_{terminals[0].id} and {terminals[-1].entity_label}_{terminals[-1].id}"
            else:
                conn_prompt = "generate routes connecting these terminals"

            # 完整的人类提示，包含：
            # - 端点位置信息
            # - 障碍物边界框信息
            # - 任务要求（生成最短无碰撞路径）
            # - 代码模板参考
            human_prompt = (
                f"The terminal positions are as follows:\n{terminals_str}\n"
                f"The bounding boxes of obstacles are as follows:\n{obstacles_str}\n"
                f"Please detect the bounding boxes of obstacles and the terminal positions first, and then {conn_prompt}, "
                f"hoping the route is shortest and collision-free with obstacles. "
                f"The reference code is as followed: {code_template}"
            )

            # 6. 生成完整的Ground Truth字符串
            # layout已经是归一化状态，to_language_string返回的是归一化后的坐标
            # 包含：障碍物 + 端点 + 路径节点
            language_string = layout.to_language_string()

            # 7. 构建对话数据
            conversation_data = {
                "conversations": [
                    {
                        "from": "human",
                        "value": human_prompt,
                    },
                    {
                        "from": "gpt",
                        "value": f"{language_string}",
                    },
                ],
                # 移除了point_clouds字段，因为仅使用文本数据训练
            }
            dataset[split].append(conversation_data)
        except Exception as e:
            print(f"Error processing scene {scene_id}: {e}")
            traceback.print_exc()
            continue

    # save train set
    print(f"Saving train set with {len(dataset['train'])} samples...")
    with open(
        os.path.join(args.dataset_dir, f"{args.dataset_name}_train.json"), "w"
    ) as f:
        json.dump(dataset["train"], f, indent=2)

    # save val set
    print(f"Saving val set with {len(dataset['val'])} samples...")
    with open(
        os.path.join(args.dataset_dir, f"{args.dataset_name}_val.json"), "w"
    ) as f:
        json.dump(dataset["val"], f, indent=2)

    # update dataset_info.json
    dataset_info = {
        f"{args.dataset_name}_train": {
            "file_name": f"{args.dataset_name}_train.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
            },
        },
        f"{args.dataset_name}_val": {
            "file_name": f"{args.dataset_name}_val.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
            },
        },
    }

    if not os.path.exists(os.path.join(args.dataset_dir, "dataset_info.json")):
        with open(os.path.join(args.dataset_dir, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)
    else:
        with open(os.path.join(args.dataset_dir, "dataset_info.json"), "r") as f:
            original_dataset_info = json.load(f)
        original_dataset_info.update(dataset_info)
        with open(os.path.join(args.dataset_dir, "dataset_info.json"), "w") as f:
            json.dump(original_dataset_info, f, indent=2)

