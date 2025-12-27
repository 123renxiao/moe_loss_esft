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
            split = df.loc[scene_id, "split"]

            # 1. Load layout content
            with open(os.path.join(layout_dir, f"{scene_id}.txt"), "r") as f:
                layout_content = f.read()

            # 2. Parse layout to objects
            layout = Layout(layout_content)
            
            # 3. Normalize and Discretize
            # 按照要求，这里进行归一化和离散化，且不进行 rotate/translate/scale/sort
            layout.normalize_and_discretize(args.num_bins)
            
            # 4. Extract Obstacles and Terminals info for Prompt
            # 此时 layout 中的实体坐标已经是归一化后的整数
            obstacles = layout.obstacles
            terminals = layout.terminals
            
            obstacles_str = "\n".join([o.to_language_string() for o in obstacles])
            terminals_str = "\n".join([t.to_language_string() for t in terminals])
            
            # 5. Construct Prompt
            if len(terminals) >= 2:
                conn_prompt = f"generate a route connecting {terminals[0].entity_label}_{terminals[0].id} and {terminals[-1].entity_label}_{terminals[-1].id}"
            else:
                conn_prompt = "generate routes connecting these terminals"

            human_prompt = (
                f"The terminal positions are as follows:\n{terminals_str}\n"
                f"The bounding boxes of obstacles are as follows:\n{obstacles_str}\n"
                f"Please detect the bounding boxes of obstacles and the terminal positions first, and then {conn_prompt}, "
                f"hoping the route is shortest and collision-free with obstacles. "
                f"The reference code is as followed: {code_template}"
            )

            # 6. Generate Full Ground Truth String
            # layout 已经是归一化状态，to_language_string 返回的就是归一化后的字符串
            language_string = layout.to_language_string()

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
                # 移除了 point_clouds 字段
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

