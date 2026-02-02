# MoE Loss ESFT - 专家专业化微调项目

本项目用于通过辅助损失函数（Auxiliary Loss）推进混合专家模型（MoE）的专家专业化训练，专注于3D空间布局理解和管道路径规划任务。

## 项目概述

本项目主要包含以下功能模块：

1. **数据集生成**：从多个数据源混合生成训练数据集
2. **空间布局处理**：解析和处理3D空间布局信息
3. **模型推理**：使用微调后的模型进行场景推理
4. **结果评估**：评估生成的管道路径是否避免碰撞

## 目录结构

```
.
├── LLaMA-Factory/              # LLaMA模型微调框架
├── Auxloss-For-Advancing-Expert-Specialization/  # 辅助损失训练脚本
├── arkitscenes-llm/            # ARKitScenes数据集相关
├── layout/                     # 空间布局处理模块
│   ├── entity.py              # 实体类定义（障碍物、端点、路径节点）
│   └── layout.py              # 布局解析和转换
├── create_llm_dataset.py      # 创建LLM训练数据集
├── mix_train_dataset_generate.py  # 混合多数据集生成训练数据
├── inference_val.py           # 验证集推理脚本
├── eval.py                    # 评估脚本（碰撞检测）
├── extract_failures.py        # 提取失败案例
└── code_template.txt          # 代码模板（数据类定义）
```

## 核心模块说明

### 1. 数据集生成

#### `mix_train_dataset_generate.py`

**功能**：从多个数据源混合生成训练数据集，支持7种不同类型的数据集。

**支持的数据集**：
- **Tag 0**: ArkitScenes - 3D场景布局数据
- **Tag 1**: MATH-500 - 数学问题求解
- **Tag 2**: GSM8K - 小学数学应用题
- **Tag 3**: MetaMathQA - 元数学问答
- **Tag 4**: Intent Recognition - 意图识别
- **Tag 5**: Law - 法律案例分析
- **Tag 6**: Text Summary - 文本摘要

**主要功能**：
- 从不同格式（HuggingFace、JSON、CSV）加载数据
- 统一转换为ShareGPT对话格式
- 为每条数据添加专家标签（_tag字段）
- 支持采样和数据打乱

**使用方法**：
```bash
python mix_train_dataset_generate.py --num_samples 500
```

**关键函数**：
- `format_sharegpt()`: 将数据转换为ShareGPT格式
- `load_hf_dataset()`: 加载HuggingFace数据集
- `load_json_dict()`: 加载JSON字典数据
- `load_csv()`: 加载CSV数据
- `load_arkitscenes()`: 加载ArkitScenes数据

#### `create_llm_dataset.py`

**功能**：将3D场景布局数据转换为LLM训练格式。

**处理流程**：
1. 读取场景布局文件（.txt格式）
2. 解析障碍物（Obstacle）和端点（Terminal）信息
3. 归一化并离散化坐标（默认640个bins）
4. 生成人类提示（包含障碍物和端点位置）
5. 生成GPT响应（包含完整的路径规划）
6. 保存为训练/验证数据集（JSON格式）

**主要参数**：
- `--dataset_dir`: 数据集目录
- `--split_csv`: 训练/验证划分文件
- `--dataset_name`: 数据集名称
- `--num_bins`: 离散化bins数量（默认640）

**使用方法**：
```bash
python create_llm_dataset.py \
  --dataset_dir arkitscenes-llm \
  --split_csv arkitscenes-llm/split.csv \
  --dataset_name arkitscenes \
  --num_bins 640
```

### 2. 空间布局处理模块

#### `layout/entity.py`

**功能**：定义3D空间实体的数据结构和操作方法。

**核心类**：

##### `Obstacle`（障碍物）
表示3D空间中的障碍物，使用两个端点和尺寸定义。

**属性**：
- `ax, ay, az`: 起点坐标
- `bx, by, bz`: 终点坐标
- `height`: 障碍物高度
- `thickness`: 障碍物厚度

**主要方法**：
- `rotate(angle)`: 绕Z轴旋转
- `translate(translation)`: 平移变换
- `scale(scaling)`: 缩放变换
- `normalize_and_discretize(num_bins)`: 归一化并离散化到[0, num_bins-1]
- `undiscretize_and_unnormalize(num_bins)`: 反归一化（恢复真实坐标）
- `to_language_string()`: 转换为文本表示，格式：`obstacle_0=Obstacle(ax,ay,az,bx,by,bz,height,thickness)`

##### `Terminal`（端点）
表示管道的起点或终点，包含位置、法向量和半径。

**属性**：
- `position_x, position_y, position_z`: 端点位置
- `normal_x, normal_y, normal_z`: 法向量（指示连接方向）
- `radius`: 端点半径

**主要方法**：
- 与Obstacle类似的变换方法
- `to_language_string()`: 格式：`terminal_0=Terminal(px,py,pz,nx,ny,nz,radius)`

##### `RouteNode`（路径节点）
表示管道路径中的节点，按ID顺序连接形成路径。

**属性**：
- `x, y, z`: 节点坐标
- `terminal_id`: 关联的端点ID（-1表示中间节点）

**主要方法**：
- `to_language_string()`: 格式：`node_0=Node(x,y,z,terminal_id)`

**归一化预设**：
```python
NORMALIZATION_PRESET = {
    "world": (0.0, 30.0),      # 世界坐标范围
    "height": (0.0, 10),       # 高度范围
    "width": (0.0, 25.6),      # 宽度范围
    "scale": (0.0, 0.5),       # 缩放范围
    "angle": (-6.2832, 6.2832) # 角度范围
}
```

#### `layout/layout.py`

**功能**：管理完整的3D场景布局，包含所有实体的集合和批量操作。

**核心类：Layout**

**属性**：
- `obstacles`: 障碍物列表
- `terminals`: 端点列表
- `route_nodes`: 路径节点列表

**主要方法**：

##### 解析和转换
- `from_str(s)`: 从文本字符串解析实体
- `to_language_string()`: 转换为文本表示
- `to_boxes()`: 转换为3D包围盒用于可视化

##### 批量变换
- `normalize_and_discretize(num_bins)`: 归一化所有实体
- `undiscretize_and_unnormalize(num_bins)`: 反归一化所有实体
- `translate(t)`: 平移所有实体
- `rotate(a)`: 旋转所有实体
- `scale(s)`: 缩放所有实体

##### 实体管理
- `get_entities()`: 获取所有实体列表
- `reorder_entities()`: 重新排序实体ID

### 3. 模型推理

#### `inference_val.py`

**功能**：使用微调后的模型对验证集进行批量推理。

**处理流程**：
1. 加载微调后的模型和分词器
2. 读取验证集场景列表
3. 对每个场景：
   - 加载布局文件
   - 归一化和离散化
   - 构建提示词（包含障碍物和端点信息）
   - 模型生成路径规划
   - 反归一化结果
   - 保存输出
4. 批量处理以提高效率

**主要参数**：
- `--model_path`: 微调模型路径
- `--layout_dir`: 布局文件目录
- `--split_csv`: 场景划分文件
- `--output_dir`: 输出目录
- `--batch_size`: 批处理大小（默认64）
- `--num_bins`: 离散化bins数量（默认640）

**提示词格式**：
```
The terminal positions are as follows:
terminal_0=Terminal(...)
terminal_1=Terminal(...)

The bounding boxes of obstacles are as follows:
obstacle_0=Obstacle(...)
obstacle_1=Obstacle(...)

Please detect the bounding boxes of obstacles and the terminal positions first, 
and then generate a route connecting terminal_0 and terminal_1, 
hoping the route is shortest and collision-free with obstacles.
The reference code is as followed: [code_template]
```

**使用方法**：
```bash
python inference_val.py \
  --model_path /path/to/model \
  --layout_dir /path/to/layouts \
  --split_csv split.csv \
  --output_dir inference_results \
  --batch_size 64
```

### 4. 结果评估

#### `eval.py`

**功能**：评估生成的管道路径是否与障碍物发生碰撞。

**碰撞检测算法**：

1. **Z轴快速排斥**：先检查线段和障碍物在Z轴是否有重叠
2. **2D投影检测**：将线段和障碍物投影到XY平面，检查是否相交
3. **3D穿透检测**：对相交部分进行精确的3D碰撞检测
4. **端点容差**：允许路径首尾节点插入障碍物一定深度（TERMINAL_TOLERANCE）

**关键配置**：
```python
EPSILON = 1e-5                    # 浮点数计算容差
TERMINAL_TOLERANCE = 0.1          # 端点容差（允许插入深度）
RESULTS_DIR = 'results_val_...'   # 推理结果目录
OUTPUT_FILE = 'evaluation_report.txt'  # 评估报告输出
```

**碰撞检测流程**：
```python
def check_segment_obstacle_collision(p1, p2, obstacle):
    # 1. Z轴范围检查
    # 2. 获取障碍物的XY平面多边形
    # 3. 处理垂直管路的特殊情况
    # 4. 检查线段与多边形的相交
    # 5. 验证3D穿透
```

**输出格式**：
```
File Name                           | Status     | Info
---------------------------------------------------------------------------------
scene_0001.txt                      | PASS       | Success
scene_0002.txt                      | FAIL       | Collision: Node 3->4 hit Obstacle 2
---------------------------------------------------------------------------------
Total: 100 | Passed: 95 | Failed: 5
Success Rate: 95.00%
```

**使用方法**：
```bash
python eval.py
```

#### `extract_failures.py`

**功能**：从多个评估报告中提取失败案例，生成对比表格。

**支持的报告**：
- eval_llm_4
- eval_llm_8
- eval_llm_10
- eval_spatiallm

**输出**：
- 控制台输出失败案例对比表
- 生成CSV文件（failures_report.csv）

**使用方法**：
```bash
python extract_failures.py
```

## 训练流程

### 1. 准备数据集

```bash
# 生成混合训练数据集
python mix_train_dataset_generate.py --num_samples 500

# 创建场景布局数据集
python create_llm_dataset.py \
  --dataset_dir arkitscenes-llm \
  --split_csv arkitscenes-llm/split.csv \
  --dataset_name arkitscenes
```

### 2. 模型微调

使用LLaMA-Factory进行微调：

```bash
export WANDB_API_KEY=<your_wandb_key>
export CUDA_VISIBLE_DEVICES=0,1

# 完整微调
llamafactory-cli train deepseek_full_sft.yaml

# 带辅助损失的微调
DS_SKIP_CUDA_CHECK=1 llamafactory-cli train deepseek_full_loss_sft.yaml
```

### 3. 推理和评估

```bash
# 在验证集上推理
python inference_val.py \
  --model_path /path/to/checkpoint \
  --output_dir inference_results

# 评估结果
python eval.py
```

## 数据格式说明

### 布局文件格式（.txt）

```
obstacle_0=Obstacle(1.5,2.3,0.0,5.2,2.3,0.0,2.5,0.3)
obstacle_1=Obstacle(3.1,4.2,0.0,3.1,8.5,0.0,3.0,0.25)
terminal_0=Terminal(1.0,1.0,1.5,0.0,1.0,0.0,0.15)
terminal_1=Terminal(9.0,9.0,1.5,0.0,-1.0,0.0,0.15)
node_0=Node(1.0,1.0,1.5,0)
node_1=Node(2.5,2.5,1.5,-1)
node_2=Node(5.0,5.0,1.5,-1)
node_3=Node(9.0,9.0,1.5,1)
```

### ShareGPT对话格式（.json）

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "问题描述"
    },
    {
      "from": "gpt",
      "value": "模型回答"
    }
  ],
  "_tag": 0
}
```

## 辅助损失训练

详细说明请参考 `Auxloss-For-Advancing-Expert-Specialization/README.md`

**关键特性**：
- 支持DeepSeek-MoE系列模型
- 通过辅助损失提升专家专业化
- 支持LoRA和完整微调
- 支持批量推理和合并

## 依赖项

主要依赖：
```
torch
transformers
pandas
numpy
scipy
shapely
tqdm
datasets
```

安装：
```bash
pip install torch transformers pandas numpy scipy shapely tqdm datasets
```

## 可视化

使用Rerun进行3D可视化：
```bash
conda activate spatillm
rerun scene0000_00.rrd
```

## 注意事项

1. **坐标系统**：使用右手坐标系，Z轴向上
2. **归一化范围**：默认world范围[0, 30]，离散化到[0, 639]
3. **端点容差**：评估时允许路径端点插入障碍物0.1单位深度
4. **批处理**：推理时使用批处理可大幅提升速度
5. **内存管理**：大模型推理需要足够的GPU内存

## 参考文献

- SceneScript: https://github.com/facebookresearch/scenescript
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Advancing Expert Specialization for Better MoE (NeurIPS 2025 oral)

## 许可证

请参考各子模块的LICENSE文件。

## 联系方式

如有问题，请提交Issue或联系项目维护者。
