import json
import torch
import os
import sys
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# Ensure we can import from the local layout directory
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
        # Set padding side to left for batch generation
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        model.eval()
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
    
    # Batch processing
    for i in tqdm(range(0, len(val_scenes), args.batch_size)):
        batch_scene_ids = val_scenes[i : i + args.batch_size]
        batch_prompts = []
        batch_valid_ids = []
        
        # Prepare batch data
        for scene_id in batch_scene_ids:
            layout_path = os.path.join(args.layout_dir, f"{scene_id}.txt")
            if not os.path.exists(layout_path):
                print(f"Warning: Layout file {layout_path} not found. Skipping.")
                continue
                
            try:
                with open(layout_path, "r") as f:
                    layout_content = f.read()

                layout = Layout(layout_content)
                layout.normalize_and_discretize(args.num_bins)
                
                obstacles = layout.obstacles
                terminals = layout.terminals
                
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
            
        # Tokenize and Generate
        try:
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in batch_prompts]
            
            model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=4096,
                    do_sample=False
                )
                
            # Decode
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Post-process and Save
            for scene_id, response in zip(batch_valid_ids, responses):
                try:
                    layout_response = Layout(response)
                    layout_response.undiscretize_and_unnormalize(args.num_bins)
                    
                    unnormalized_response_lines = []
                    for obs in layout_response.obstacles:
                        unnormalized_response_lines.append(obs.to_language_string())
                    for term in layout_response.terminals:
                        unnormalized_response_lines.append(term.to_language_string())
                    for node in layout_response.route_nodes:
                        unnormalized_response_lines.append(node.to_language_string())
                        
                    unnormalized_response = "\n".join(unnormalized_response_lines)
                    
                    output_filename = os.path.join(args.output_dir, f"{scene_id}.txt")
                    with open(output_filename, "w") as f:
                        f.write(unnormalized_response)
                        
                except Exception as e:
                    print(f"Error parsing response for scene {scene_id}: {e}")
                    # Save raw response for debugging
                    with open(os.path.join(args.output_dir, f"{scene_id}_error.txt"), "w") as f:
                        f.write(response)

        except Exception as e:
            print(f"Error during batch generation: {e}")
            continue

    print(f"All results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
