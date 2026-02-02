# 文档索引

本项目现已包含完整的中文文档说明。以下是各文档的说明和使用指南。

## 📚 文档列表

### 1. [README.md](README.md) - 项目主文档
**适合人群**：所有用户

**内容概览**：
- 项目整体介绍和功能概述
- 目录结构说明
- 核心模块功能说明
- 训练和评估流程
- 数据格式规范
- 依赖项安装

**何时阅读**：
- 首次接触项目时
- 需要了解项目整体架构时
- 查找特定模块功能时

### 2. [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) - 使用教程
**适合人群**：实际使用者、开发者

**内容概览**：
- 环境配置步骤
- 数据准备详细教程
- 模型训练配置和命令
- 推理脚本使用方法
- 评估工具使用
- 常见问题解答（FAQ）
- 进阶技巧

**何时阅读**：
- 开始使用项目时
- 遇到具体操作问题时
- 需要调优或定制功能时

### 3. [LAYOUT_MODULE_DOC.md](LAYOUT_MODULE_DOC.md) - Layout模块技术文档
**适合人群**：深度开发者、研究人员

**内容概览**：
- 实体类详细设计（Obstacle、Terminal、RouteNode）
- 坐标变换算法（归一化、离散化）
- 数学公式和精度分析
- 几何计算原理
- 代码示例

**何时阅读**：
- 需要理解底层实现时
- 修改或扩展实体类时
- 调试坐标变换问题时
- 进行算法研究时

## 🎯 快速导航

### 我想...

#### 了解这个项目是做什么的
👉 阅读 [README.md](README.md) 的"项目概述"部分

#### 开始使用这个项目
👉 阅读 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) 的"快速开始"部分

#### 准备训练数据
👉 阅读 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) 的"数据准备"部分

#### 训练模型
👉 阅读 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) 的"模型训练"部分

#### 使用模型进行推理
👉 阅读 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) 的"模型推理"部分

#### 评估结果质量
👉 阅读 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) 的"结果评估"部分

#### 理解数据格式
👉 阅读 [README.md](README.md) 的"数据格式说明"部分

#### 理解坐标变换
👉 阅读 [LAYOUT_MODULE_DOC.md](LAYOUT_MODULE_DOC.md) 的"坐标变换详解"部分

#### 扩展实体类
👉 阅读 [LAYOUT_MODULE_DOC.md](LAYOUT_MODULE_DOC.md) 的"entity.py"部分

#### 解决遇到的问题
👉 阅读 [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) 的"常见问题"部分

## 📖 推荐阅读路径

### 路径1：新用户快速上手
1. README.md - 项目概述（5分钟）
2. README.md - 目录结构（3分钟）
3. USAGE_EXAMPLES.md - 快速开始（10分钟）
4. USAGE_EXAMPLES.md - 数据准备（15分钟）
5. 开始实践！

### 路径2：研究人员深入理解
1. README.md - 完整阅读（20分钟）
2. LAYOUT_MODULE_DOC.md - 实体类设计（30分钟）
3. LAYOUT_MODULE_DOC.md - 坐标变换详解（20分钟）
4. 阅读源代码中的注释
5. USAGE_EXAMPLES.md - 进阶技巧（15分钟）

### 路径3：开发者定制功能
1. README.md - 核心模块说明（15分钟）
2. LAYOUT_MODULE_DOC.md - 相关模块详解（30分钟）
3. 阅读对应源代码文件
4. USAGE_EXAMPLES.md - 相关示例（10分钟）
5. 开始开发！

## 📝 源代码注释

以下源代码文件已添加详细的中文注释：

### 数据处理
- `mix_train_dataset_generate.py` - 混合数据集生成
  - 各数据集加载函数的说明
  - ShareGPT格式转换逻辑
  - 采样和打乱算法

- `create_llm_dataset.py` - LLM训练数据生成
  - 布局文件解析流程
  - 提示词构建逻辑
  - 归一化和离散化说明

### 模型推理
- `inference_val.py` - 批量推理脚本
  - 模型加载和配置
  - 批处理优化说明
  - 输出后处理流程

### 结果评估
- `eval.py` - 碰撞检测评估
  - 碰撞检测算法详解
  - Z轴快速排斥原理
  - 3D穿透检测逻辑
  - 端点容差处理

### 核心模块
- `layout/entity.py` - 实体类定义
  - 各实体类的属性说明
  - 坐标变换方法注释
  - 归一化算法说明

- `layout/layout.py` - 布局管理
  - 解析和转换方法
  - 批量操作实现
  - 包围盒生成逻辑

## 💡 使用建议

### 查阅文档的最佳实践

1. **先看目录**：每个文档都有详细目录，快速定位需要的内容
2. **善用搜索**：在文档中搜索关键词（如"归一化"、"碰撞检测"等）
3. **结合实践**：边看文档边运行代码，理解更深刻
4. **参考示例**：文档中的代码示例可以直接复制使用
5. **记录问题**：遇到文档未覆盖的问题，可以提Issue

### 文档更新

本文档会随项目持续更新。主要更新内容包括：
- 新功能的使用说明
- 更多实用示例
- 问题解答
- 性能优化建议

## 🔗 外部资源

### 相关项目文档
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - 模型微调框架
- [Auxloss-For-Advancing-Expert-Specialization](./Auxloss-For-Advancing-Expert-Specialization/README.md) - 辅助损失训练

### 技术参考
- [SceneScript](https://github.com/facebookresearch/scenescript) - 场景脚本语言
- [Shapely](https://shapely.readthedocs.io/) - 几何计算库
- [Transformers](https://huggingface.co/docs/transformers/) - HuggingFace模型库

## 📮 反馈和贡献

### 如何反馈问题
- 文档不清楚的地方
- 发现的错误或疏漏
- 需要补充的内容
- 使用中的困惑

请通过以下方式反馈：
1. 提交GitHub Issue
2. 提交Pull Request改进文档
3. 联系项目维护者

### 文档贡献指南

欢迎贡献文档！建议：
1. 保持中文表述清晰准确
2. 添加实用的代码示例
3. 注明适用场景和限制
4. 保持格式统一

---

**文档版本**：v1.0  
**最后更新**：2026年2月  
**维护者**：项目团队

感谢使用本项目！希望这些文档能帮助您快速上手和深入理解。
