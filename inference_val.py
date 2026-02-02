"""
验证集推理脚本
===============
使用微调后的LLM模型对验证集场景进行批量推理，生成管道路径规划结果。

主要功能：
1. 加载微调后的模型和分词器
2. 批量处理验证集场景
3. 构建包含障碍物和端点信息的提示词
4. 生成路径规划（归一化坐标）
5. 反归一化并保存结果

性能优化：
- 使用批处理提高推理速度
- 左侧padding以支持批量生成
- 使用bfloat16降低显存占用

作者：Li Wentian
日期：2025年
"""

import json
import torch
import os
import sys
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# 确保可以从本地layout目录导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from layout.layout import Layout

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/storage2/liwentian/llm_pipe_gen/LLaMA-Factory/saves/qwen3-4b/10_epoch/sft/checkpoint-2000")
    parser.add_argument("--layout_dir", type=str, default="/data/storage2/liwentian/SpatialLM_pipe_gen/arkitscenes-spatiallm/layout")
    parser.add_argument("--split_csv", type=str, default="/data/storage2/liwentian/SpatialLM_pipe_gen/arkitscenes-spatiallm/split.csv")
    parser.add_argument("--code_template", type=str, default="code_template.txt")
    parser.add_argument("--output_dir", type=str, default="inference_results_4b_4")
    parser.add_argument("--num_bins", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载模型和分词器
    print(f"Loading model from {args.model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        # 设置padding方向为左侧，这对批量生成很重要
        # 左侧padding确保所有序列的最后一个token对齐，便于并行生成
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # 加载模型，使用bfloat16以节省显存
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            device_map="auto",  # 自动分配到可用GPU
            torch_dtype=torch.bfloat16,  # 使用bfloat16混合精度
            trust_remote_code=True
        )
        model.eval()  # 设置为评估模式
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 读取数据
    print(f"Loading split from {args.split_csv}...")
    if not os.path.exists(args.split_csv):
        print(f"Error: {args.split_csv} not found.")
        return

    df = pd.read_csv(args.split_csv, dtype=str)
    # Filter for validation set
    val_scenes = df[df['split'] == 'val']['id'].tolist()
    print(f"Found {len(val_scenes)} validation scenes.")

    # Load code template
    code_template_file = args.code_template
    if not os.path.exists(code_template_file):
         # Try absolute path if relative fails
         script_dir = os.path.dirname(os.path.abspath(__file__))
         code_template_file_abs = os.path.join(script_dir, code_template_file)
         if os.path.exists(code_template_file_abs):
             code_template_file = code_template_file_abs
         elif os.path.exists("/data/storage2/liwentian/SpatialLM_pipe_gen/code_template.txt"):
             code_template_file = "/data/storage2/liwentian/SpatialLM_pipe_gen/code_template.txt"
             print(f"Using fallback code template: {code_template_file}")
         else:
             print(f"Error: {code_template_file} not found.")
             return

    with open(code_template_file, "r") as f:
        code_template = f.read()

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # 3. 推理循环
    print(f"Starting inference on {len(val_scenes)} samples with batch size {args.batch_size}...")
    
    # 批处理：将验证集分成多个batch
    for i in tqdm(range(0, len(val_scenes), args.batch_size)):
        batch_scene_ids = val_scenes[i : i + args.batch_size]
        batch_prompts = []
        batch_valid_ids = []
        
        # 准备batch数据
        for scene_id in batch_scene_ids:
            layout_path = os.path.join(args.layout_dir, f"{scene_id}.txt")
            if not os.path.exists(layout_path):
                print(f"Warning: Layout file {layout_path} not found. Skipping.")
                continue
                
            try:
                # 读取并解析布局文件
                with open(layout_path, "r") as f:
                    layout_content = f.read()

                layout = Layout(layout_content)
                # 归一化并离散化，与训练时保持一致
                layout.normalize_and_discretize(args.num_bins)
                
                obstacles = layout.obstacles
                terminals = layout.terminals
                
                # 构建提示词
                obstacles_str = "\n".join([o.to_language_string() for o in obstacles])
                terminals_str = "\n".join([t.to_language_string() for t in terminals])
                
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
                
                batch_prompts.append(human_prompt)
                batch_valid_ids.append(scene_id)
                
            except Exception as e:
                print(f"Error preparing scene {scene_id}: {e}")
                continue
        
        if not batch_prompts:
            continue
            
        # 批量Tokenize和生成
        try:
            # 应用对话模板
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in batch_prompts]
            
            # Tokenize，使用padding和truncation
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            # 生成路径规划结果
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=4096,  # 允许生成长路径
                    do_sample=False  # 使用贪婪解码，确保结果确定性
                )
                
            # 解码：只保留新生成的token
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # 后处理并保存
            for scene_id, response in zip(batch_valid_ids, responses):
                try:
                    # 解析生成的文本为Layout对象
                    layout_response = Layout(response)
                    # 反归一化，恢复真实世界坐标
                    layout_response.undiscretize_and_unnormalize(args.num_bins)
                    
                    # 将所有实体转换为文本行
                    unnormalized_response_lines = []
                    for obs in layout_response.obstacles:
                        unnormalized_response_lines.append(obs.to_language_string())
                    for term in layout_response.terminals:
                        unnormalized_response_lines.append(term.to_language_string())
                    for node in layout_response.route_nodes:
                        unnormalized_response_lines.append(node.to_language_string())
                        
                    unnormalized_response = "\n".join(unnormalized_response_lines)
                    
                    # 保存到文件
                    output_filename = os.path.join(args.output_dir, f"{scene_id}.txt")
                    with open(output_filename, "w") as f:
                        f.write(unnormalized_response)
                        
                except Exception as e:
                    print(f"Error parsing response for scene {scene_id}: {e}")
                    # 保存原始响应以便调试
                    with open(os.path.join(args.output_dir, f"{scene_id}_error.txt"), "w") as f:
                        f.write(response)

        except Exception as e:
            print(f"Error during batch generation: {e}")
            continue

    print(f"All results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
