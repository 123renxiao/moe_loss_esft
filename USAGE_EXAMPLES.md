# 使用示例和教程

本文档提供详细的使用示例，帮助您快速上手MoE Loss ESFT项目。

## 目录
1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [模型推理](#模型推理)
5. [结果评估](#结果评估)
6. [常见问题](#常见问题)

## 快速开始

### 环境配置

```bash
# 创建conda环境
conda create -n spatiallm python=3.10
conda activate spatiallm

# 安装依赖
pip install torch transformers pandas numpy scipy shapely tqdm datasets
pip install -e . --no-build-isolation
```

### 最小示例

```bash
# 1. 准备数据
python create_llm_dataset.py \
  --dataset_dir ./data \
  --split_csv ./data/split.csv \
  --dataset_name demo

# 2. 训练（使用LLaMA-Factory）
llamafactory-cli train config.yaml

# 3. 推理
python inference_val.py \
  --model_path ./output/model \
  --output_dir ./results

# 4. 评估
python eval.py
```

## 数据准备

### 1. 准备3D场景布局数据

布局文件格式示例 (`scene_001.txt`):

```
obstacle_0=Obstacle(1.5,2.3,0.0,5.2,2.3,0.0,2.5,0.3)
obstacle_1=Obstacle(3.1,4.2,0.0,3.1,8.5,0.0,3.0,0.25)
terminal_0=Terminal(1.0,1.0,1.5,0.0,1.0,0.0,0.15)
terminal_1=Terminal(9.0,9.0,1.5,0.0,-1.0,0.0,0.15)
```

**字段说明**:
- `Obstacle(ax,ay,az,bx,by,bz,height,thickness)`: 障碍物
  - `(ax,ay,az)`: 起点坐标
  - `(bx,by,bz)`: 终点坐标
  - `height`: 高度
  - `thickness`: 厚度

- `Terminal(px,py,pz,nx,ny,nz,radius)`: 端点
  - `(px,py,pz)`: 位置坐标
  - `(nx,ny,nz)`: 法向量
  - `radius`: 半径

### 2. 准备split.csv文件

创建训练/验证划分文件:

```csv
id,split
scene_001,train
scene_002,train
scene_003,val
scene_004,val
```

### 3. 生成训练数据集

#### 3.1 单一数据集（场景布局）

```bash
python create_llm_dataset.py \
  --dataset_dir ./arkitscenes-llm \
  --split_csv ./arkitscenes-llm/split.csv \
  --dataset_name arkitscenes \
  --num_bins 640
```

**输出**:
- `arkitscenes_train.json`: 训练集
- `arkitscenes_val.json`: 验证集
- `dataset_info.json`: 数据集配置信息

#### 3.2 混合多数据集

```bash
python mix_train_dataset_generate.py --num_samples 1000
```

这将从7个不同领域的数据集中各采样1000条数据，生成混合训练集。

**支持的数据集**:

| Tag | 数据集名称 | 领域 | 用途 |
|-----|-----------|------|------|
| 0 | ArkitScenes | 3D场景 | 空间理解 |
| 1 | MATH-500 | 数学 | 推理能力 |
| 2 | GSM8K | 应用题 | 逻辑推理 |
| 3 | MetaMathQA | 数学问答 | 解题能力 |
| 4 | Intent Recognition | 意图识别 | 语义理解 |
| 5 | Law | 法律 | 文本分析 |
| 6 | Text Summary | 摘要 | 生成能力 |

## 模型训练

### 1. 准备训练配置

创建 `deepseek_full_sft.yaml`:

```yaml
### model
model_name_or_path: deepseek-ai/deepseek-moe-16b-chat

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: examples/deepspeed/ds_z3_config.json

### dataset
dataset: arkitscenes_train
template: deepseek
cutoff_len: 4096
max_samples: 10000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: saves/arkitscenes/full
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### 2. 开始训练

#### 完整微调

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_API_KEY=your_wandb_key

llamafactory-cli train deepseek_full_sft.yaml
```

#### LoRA微调（节省资源）

修改配置文件:

```yaml
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
lora_target: all
```

然后运行:

```bash
llamafactory-cli train deepseek_lora_sft.yaml
```

### 3. 带辅助损失的训练

```bash
DS_SKIP_CUDA_CHECK=1 llamafactory-cli train deepseek_full_loss_sft.yaml
```

**辅助损失的作用**:
- 促进专家专业化
- 提高多任务学习效果
- 改善路由分配

### 4. 监控训练

使用Weights & Biases监控:

```bash
# 查看训练曲线
wandb login
# 访问 https://wandb.ai/your-username/your-project
```

使用TensorBoard监控:

```bash
tensorboard --logdir saves/arkitscenes/full
```

## 模型推理

### 1. 基本推理

```bash
python inference_val.py \
  --model_path ./saves/arkitscenes/full/checkpoint-2000 \
  --layout_dir ./data/layout \
  --split_csv ./data/split.csv \
  --output_dir ./inference_results \
  --batch_size 32 \
  --num_bins 640
```

**参数说明**:
- `--model_path`: 模型检查点路径
- `--layout_dir`: 布局文件目录
- `--split_csv`: 场景划分文件
- `--output_dir`: 输出目录
- `--batch_size`: 批处理大小（根据GPU内存调整）
- `--num_bins`: 离散化bins数量（必须与训练时一致）

### 2. LoRA模型推理

如果使用LoRA训练，需要先合并:

```bash
# 使用LLaMA-Factory合并
llamafactory-cli export \
  --model_name_or_path deepseek-ai/deepseek-moe-16b-chat \
  --adapter_name_or_path ./saves/arkitscenes/lora \
  --export_dir ./saves/arkitscenes/merged \
  --export_size 2 \
  --export_device cpu

# 然后使用合并后的模型推理
python inference_val.py \
  --model_path ./saves/arkitscenes/merged \
  --output_dir ./inference_results
```

### 3. 批量推理脚本

创建 `batch_inference.sh`:

```bash
#!/bin/bash

CHECKPOINTS=(
  "./saves/arkitscenes/full/checkpoint-1000"
  "./saves/arkitscenes/full/checkpoint-2000"
  "./saves/arkitscenes/full/checkpoint-3000"
)

for ckpt in "${CHECKPOINTS[@]}"; do
  echo "Processing $ckpt"
  python inference_val.py \
    --model_path "$ckpt" \
    --output_dir "./results/$(basename $ckpt)" \
    --batch_size 64
done
```

## 结果评估

### 1. 碰撞检测评估

编辑 `eval.py` 配置:

```python
RESULTS_DIR = './inference_results'
OUTPUT_FILE = 'evaluation_report.txt'
TERMINAL_TOLERANCE = 0.1  # 端点容差
```

运行评估:

```bash
python eval.py
```

**输出示例**:

```
File Name                           | Status     | Info
---------------------------------------------------------------------------------
scene_001.txt                       | PASS       | Success
scene_002.txt                       | FAIL       | Collision: Node 3->4 hit Obstacle 2
scene_003.txt                       | PASS       | Success
---------------------------------------------------------------------------------
Total: 100 | Passed: 95 | Failed: 5
Success Rate: 95.00%

List of Failed Files:
 - scene_002.txt
 - scene_007.txt
 - scene_015.txt
 - scene_042.txt
 - scene_088.txt
```

### 2. 对比多个模型

```bash
# 评估多个检查点
for ckpt in checkpoint-1000 checkpoint-2000 checkpoint-3000; do
  python eval.py \
    --results_dir ./results/$ckpt \
    --output_file evaluation_${ckpt}.txt
done

# 提取失败案例对比
python extract_failures.py
```

### 3. 可视化结果

使用Rerun进行3D可视化:

```bash
# 安装rerun
pip install rerun-sdk

# 可视化场景
rerun scene_001.rrd
```

## 常见问题

### Q1: 如何调整离散化bins数量？

**A**: 修改 `num_bins` 参数。更大的bins提供更高精度，但需要更多计算资源。

```bash
# 训练时
python create_llm_dataset.py --num_bins 1024

# 推理时（必须保持一致）
python inference_val.py --num_bins 1024
```

### Q2: GPU内存不足怎么办？

**A**: 尝试以下方法:

1. 减小batch_size:
```bash
python inference_val.py --batch_size 8
```

2. 使用LoRA而非完整微调

3. 使用梯度检查点:
```yaml
gradient_checkpointing: true
```

4. 使用DeepSpeed ZeRO-3:
```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
```

### Q3: 如何添加自定义数据集？

**A**: 修改 `mix_train_dataset_generate.py`:

```python
# 添加新的加载函数
def load_my_dataset(path, num_samples, tag_id):
    # 加载逻辑
    data = load_data(path)
    # 转换为ShareGPT格式
    formatted = [format_sharegpt(item['input'], item['output'], tag_id) for item in data]
    return formatted

# 在process_datasets中调用
def process_datasets(args):
    combined_data = []
    # ... 现有数据集 ...
    
    # 添加新数据集 (Tag 7)
    combined_data.extend(load_my_dataset(my_dataset_path, args.num_samples, 7))
```

### Q4: 评估时端点容差如何设置？

**A**: 端点容差允许路径首尾插入障碍物一定深度。根据实际场景调整:

```python
# eval.py
TERMINAL_TOLERANCE = 0.1  # 默认0.1单位

# 如果端点需要穿墙更深
TERMINAL_TOLERANCE = 0.2

# 如果不允许任何插入
TERMINAL_TOLERANCE = 0.0
```

### Q5: 如何处理训练数据不平衡？

**A**: 使用采样控制各数据集比例:

```bash
# 为每个数据集设置不同的采样数
python mix_train_dataset_generate.py --num_samples 500

# 或修改代码为每个数据集单独设置
combined_data.extend(load_arkitscenes(path, 2000, 0))  # 更多场景数据
combined_data.extend(load_hf_dataset(path, 'train', 500, 1, ...))  # 较少数学数据
```

### Q6: 推理速度太慢怎么办？

**A**: 优化建议:

1. 增大batch_size（如果GPU内存允许）
2. 使用vLLM加速:
```bash
# 参考 Auxloss-For-Advancing-Expert-Specialization/Scripts/Inference/batch_vllm_infer.py
```

3. 使用量化模型:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True  # 或 load_in_4bit=True
)
```

### Q7: 如何调试模型生成结果？

**A**: 查看详细输出:

```python
# 在inference_val.py中添加调试输出
print(f"Prompt:\n{human_prompt}\n")
print(f"Response:\n{response}\n")
print(f"Parsed layout:\n{layout_response.to_language_string()}\n")
```

对于解析失败的场景，会自动保存到 `{scene_id}_error.txt`。

## 进阶技巧

### 1. 自定义提示词模板

修改 `create_llm_dataset.py` 中的提示词:

```python
human_prompt = (
    f"场景描述：\n"
    f"端点位置：\n{terminals_str}\n"
    f"障碍物：\n{obstacles_str}\n"
    f"任务：生成从{terminals[0]}到{terminals[-1]}的最短无碰撞路径\n"
    f"要求：\n"
    f"1. 路径必须避开所有障碍物\n"
    f"2. 尽可能短\n"
    f"3. 使用以下格式输出：{code_template}"
)
```

### 2. 多GPU并行推理

```bash
# 使用torchrun
torchrun --nproc_per_node=4 inference_val.py \
  --model_path ./model \
  --output_dir ./results
```

### 3. 增量训练

```bash
# 从检查点继续训练
llamafactory-cli train config.yaml \
  --resume_from_checkpoint ./saves/checkpoint-1000
```

## 相关资源

- [LLaMA-Factory文档](https://github.com/hiyouga/LLaMA-Factory)
- [DeepSeek模型](https://huggingface.co/deepseek-ai)
- [SceneScript项目](https://github.com/facebookresearch/scenescript)
- [Shapely文档](https://shapely.readthedocs.io/)

## 获取帮助

如遇问题，请：
1. 检查本文档的常见问题部分
2. 查看项目README
3. 提交GitHub Issue
4. 联系项目维护者

祝使用愉快！
