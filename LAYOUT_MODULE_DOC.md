# Layout模块详细说明

本文档详细说明layout模块的设计和实现，包含实体类、坐标变换和格式转换等核心功能。

## 模块结构

```
layout/
├── __init__.py          # 模块初始化
├── entity.py           # 实体类定义（Obstacle, Terminal, RouteNode）
└── layout.py           # 布局管理类
```

## entity.py - 实体类

### 归一化预设

所有实体使用统一的归一化范围：

```python
NORMALIZATION_PRESET = {
    "world": (0.0, 30.0),      # 世界坐标范围 [0, 30]米
    "height": (0.0, 10),       # 高度范围 [0, 10]米
    "width": (0.0, 25.6),      # 宽度范围 [0, 25.6]米
    "scale": (0.0, 0.5),       # 缩放范围 [0, 0.5]
    "angle": (-6.2832, 6.2832) # 角度范围 [-2π, 2π]弧度
}
```

### Obstacle类（障碍物）

#### 数据结构

```python
@dataclass
class Obstacle:
    id: int                    # 障碍物唯一标识
    ax, ay, az: float         # 起点A坐标
    bx, by, bz: float         # 终点B坐标
    height: float             # 障碍物高度（Z方向）
    thickness: float          # 障碍物厚度（垂直AB方向）
    entity_label: str = "obstacle"
```

#### 几何表示

障碍物是一个3D长方体，由以下参数定义：

1. **中心线**：连接点A(ax, ay, az)和点B(bx, by, bz)
2. **长度**：|AB|的长度
3. **厚度**：垂直于AB方向的宽度
4. **高度**：Z方向的高度

**XY平面投影**：
```
         thickness
      <------------>
      
  A +--------------+ B
    |              |
    |   中心线AB   |
    |              |
  A'+--------------+ B'
```

#### 主要方法

##### 1. rotate(angle)
绕Z轴旋转障碍物

```python
def rotate(self, angle: float):
    """
    参数:
        angle: 旋转角度（弧度），逆时针为正
    
    算法:
        使用旋转矩阵 R(θ) 变换A和B点的坐标
    """
    rotmat = R.from_rotvec([0, 0, angle]).as_matrix()
    # 变换起点和终点
```

##### 2. translate(translation)
平移障碍物

```python
def translate(self, translation: np.ndarray):
    """
    参数:
        translation: [dx, dy, dz] 平移向量
    
    效果:
        A' = A + translation
        B' = B + translation
    """
```

##### 3. scale(scaling)
缩放障碍物

```python
def scale(self, scaling: float):
    """
    参数:
        scaling: 缩放因子（>1放大，<1缩小）
    
    缩放内容:
        - 所有坐标值 (ax, ay, az, bx, by, bz)
        - 高度 (height)
        - 厚度 (thickness)
    """
```

##### 4. normalize_and_discretize(num_bins)
归一化并离散化坐标

```python
def normalize_and_discretize(self, num_bins):
    """
    参数:
        num_bins: 离散化bins数量（如640）
    
    步骤:
        1. 归一化：将真实坐标映射到[0, 1]
           normalized = (value - min) / (max - min)
        
        2. 离散化：映射到整数bins
           discretized = int(normalized * num_bins)
        
        3. 裁剪：确保在[0, num_bins-1]范围内
           clipped = np.clip(discretized, 0, num_bins-1)
    
    示例:
        输入: ax = 15.0, world范围[0, 30], num_bins=640
        归一化: (15.0 - 0.0) / (30.0 - 0.0) = 0.5
        离散化: 0.5 * 640 = 320
        输出: ax = 320
    """
```

##### 5. undiscretize_and_unnormalize(num_bins)
反归一化（恢复真实坐标）

```python
def undiscretize_and_unnormalize(self, num_bins):
    """
    参数:
        num_bins: 离散化bins数量（必须与离散化时相同）
    
    步骤:
        1. 反离散化：整数bins映射回[0, 1]
           undiscretized = discretized / num_bins
        
        2. 反归一化：映射回真实坐标范围
           real_value = undiscretized * (max - min) + min
    
    示例:
        输入: ax = 320, num_bins=640, world范围[0, 30]
        反离散化: 320 / 640 = 0.5
        反归一化: 0.5 * (30.0 - 0.0) + 0.0 = 15.0
        输出: ax = 15.0
    """
```

##### 6. to_language_string()
转换为文本表示

```python
def to_language_string(self):
    """
    返回:
        格式化字符串 "obstacle_{id}=Obstacle(ax,ay,az,bx,by,bz,height,thickness)"
    
    示例:
        "obstacle_0=Obstacle(1.5,2.3,0.0,5.2,2.3,0.0,2.5,0.3)"
    """
```

### Terminal类（端点）

#### 数据结构

```python
@dataclass
class Terminal:
    id: int                              # 端点唯一标识
    position_x, position_y, position_z: float  # 端点位置
    normal_x, normal_y, normal_z: float        # 法向量（连接方向）
    radius: float                        # 端点半径
    entity_label: str = "terminal"
```

#### 几何意义

端点表示管道的起点或终点：

```
         法向量 (nx, ny, nz)
              ↑
              |
         +----o----+
         |  (位置) |  半径r
         +---------+
```

#### 特殊方法

##### rotate(angle)
旋转端点（包括位置和法向量）

```python
def rotate(self, angle: float):
    """
    与Obstacle不同，Terminal的法向量也需要旋转
    
    步骤:
        1. 旋转位置坐标
        2. 旋转法向量（重要！）
    
    为什么需要旋转法向量？
        法向量指示连接方向，旋转场景时方向也应该跟着转
    """
```

##### normalize_and_discretize(num_bins)
端点的归一化有特殊处理

```python
def normalize_and_discretize(self, num_bins):
    """
    特殊之处:
        1. 位置坐标使用world范围 [0, 30]
        2. 法向量使用[-1, 1]范围（因为是单位向量）
        3. 半径使用scale范围 [0, 0.5]
    
    法向量归一化示例:
        输入: normal_x = 0.707 (单位向量的一个分量)
        范围: [-1, 1]
        归一化: (0.707 - (-1)) / (1 - (-1)) = 0.8535
        离散化: 0.8535 * 640 = 546.24 ≈ 546
        输出: normal_x = 546
    """
```

##### to_language_string()
转换为文本表示

```python
def to_language_string(self):
    """
    返回:
        "terminal_{id}=Terminal(px,py,pz,nx,ny,nz,radius)"
    
    注意:
        ID会取模1000，防止ID过大
        self.id = self.id % 1000
    """
```

### RouteNode类（路径节点）

#### 数据结构

```python
@dataclass
class RouteNode:
    id: int                    # 节点序号（决定连接顺序）
    x, y, z: float            # 节点坐标
    terminal_id: int = -1     # 关联的端点ID（-1表示中间节点）
    entity_label: str = "node"
```

#### 路径表示

路径由一系列按ID排序的节点组成：

```
Terminal_0 -> Node_0 -> Node_1 -> Node_2 -> ... -> Node_N -> Terminal_1
   (起点)    (路径点)  (路径点)   (路径点)         (路径点)    (终点)
```

**连接规则**：
- Node按ID从小到大连接
- Node_i 连接到 Node_{i+1}
- 第一个Node通常与起始Terminal关联（terminal_id=0）
- 最后一个Node通常与目标Terminal关联（terminal_id=1）

#### 主要方法

##### to_language_string()

```python
def to_language_string(self):
    """
    返回:
        "node_{id}=Node(x,y,z,terminal_id)"
    
    示例:
        起始节点: "node_0=Node(1.0,1.0,1.5,0)"
        中间节点: "node_1=Node(2.5,2.5,1.5,-1)"
        终止节点: "node_5=Node(9.0,9.0,1.5,1)"
    """
```

## layout.py - 布局管理类

### Layout类

#### 数据结构

```python
class Layout:
    def __init__(self, s: str = None):
        self.obstacles = []      # Obstacle对象列表
        self.terminals = []      # Terminal对象列表
        self.route_nodes = []    # RouteNode对象列表
```

#### 主要方法

##### 1. from_str(s)
从文本字符串解析布局

```python
def from_str(self, s: str):
    """
    参数:
        s: 包含多行实体定义的字符串
    
    解析规则:
        - 每行格式: "entity_label_id=Type(params)"
        - 根据entity_label分发到对应类
        - 跳过格式错误的行
    
    示例输入:
        obstacle_0=Obstacle(1.5,2.3,0.0,5.2,2.3,0.0,2.5,0.3)
        terminal_0=Terminal(1.0,1.0,1.5,0.0,1.0,0.0,0.15)
        node_0=Node(1.0,1.0,1.5,0)
    
    解析流程:
        1. 按行分割
        2. 查找"="和"("
        3. 提取entity_label和id
        4. 提取参数列表
        5. 创建对应实体对象
    """
```

##### 2. to_language_string()
转换为文本字符串

```python
def to_language_string(self):
    """
    返回:
        所有实体的文本表示，按类型分组
    
    顺序:
        1. 所有障碍物
        2. 所有端点
        3. 所有路径节点
    
    示例输出:
        obstacle_0=Obstacle(...)
        obstacle_1=Obstacle(...)
        terminal_0=Terminal(...)
        terminal_1=Terminal(...)
        node_0=Node(...)
        node_1=Node(...)
    """
```

##### 3. normalize_and_discretize(num_bins)
批量归一化和离散化

```python
def normalize_and_discretize(self, num_bins):
    """
    对所有实体执行归一化和离散化
    
    效果:
        - 真实世界坐标 -> 整数bins
        - 便于模型以token方式表示坐标
    """
```

##### 4. undiscretize_and_unnormalize(num_bins)
批量反归一化

```python
def undiscretize_and_unnormalize(self, num_bins):
    """
    对所有实体执行反归一化
    
    效果:
        - 整数bins -> 真实世界坐标
        - 用于推理后恢复真实坐标
    """
```

##### 5. to_boxes()
转换为3D包围盒（用于可视化）

```python
def to_boxes(self):
    """
    返回:
        包围盒列表，每个包围盒包含:
        - id: 唯一标识
        - class: 类型（Obstacle/Terminal/Route）
        - label: 显示标签
        - center: 中心坐标
        - rotation: 旋转矩阵
        - scale: 尺寸 [长, 宽, 高]
    
    用途:
        - 3D可视化（Rerun等）
        - 碰撞检测
        - 调试和验证
    
    包围盒生成规则:
        
        Obstacle:
            - center = (A + B) / 2，Z方向加上height/2
            - rotation = 沿AB方向的旋转矩阵
            - scale = [length, thickness, height]
        
        Terminal:
            - center = position
            - rotation = 单位矩阵
            - scale = [2*radius, 2*radius, 2*radius]
        
        RouteNode (段):
            - center = (Node_i + Node_{i+1}) / 2
            - rotation = 沿节点连线的3D旋转矩阵
            - scale = [segment_length, 0.1, 0.1]
    """
```

##### 6. reorder_entities()
重新排序实体ID

```python
def reorder_entities(self):
    """
    重新分配实体ID，确保连续性和一致性
    
    排序规则:
        - Obstacles: 按(ax, ay)空间位置排序
        - Terminals: 按(position_x, position_y)空间位置排序
        - RouteNodes: 严格按ID排序（保持路径顺序）
    
    ID重建:
        排序后重新分配ID为 0, 1, 2, ...
    """
```

## 坐标变换详解

### 归一化和离散化的必要性

#### 为什么需要归一化？

1. **统一尺度**：不同场景的坐标范围差异大，归一化到[0, 1]统一处理
2. **数值稳定性**：避免过大或过小的数值导致训练不稳定
3. **模型泛化**：模型学习相对位置而非绝对坐标

#### 为什么需要离散化？

1. **Token表示**：LLM使用离散token，需要将连续坐标映射到有限词表
2. **精度控制**：通过bins数量控制精度（640 bins ≈ 5cm精度，对于30m场景）
3. **序列化**：便于文本生成和解析

### 变换流程示例

#### 场景：30m×30m房间，障碍物在(15, 20, 0)到(25, 20, 0)

**原始数据**：
```python
Obstacle(
    ax=15.0, ay=20.0, az=0.0,
    bx=25.0, by=20.0, bz=0.0,
    height=2.5, thickness=0.3
)
```

**归一化（world范围[0, 30]，num_bins=640）**：

```python
# ax归一化
normalized_ax = (15.0 - 0.0) / (30.0 - 0.0) = 0.5

# 离散化
discretized_ax = int(0.5 * 640) = 320

# 裁剪
ax = clip(320, 0, 639) = 320
```

**结果**：
```python
Obstacle(
    ax=320, ay=427, az=0,
    bx=533, by=427, bz=0,
    height=160, thickness=19
)
```

**反归一化**：

```python
# 反离散化
undiscretized_ax = 320 / 640 = 0.5

# 反归一化
real_ax = 0.5 * (30.0 - 0.0) + 0.0 = 15.0
```

**恢复原始值**：
```python
Obstacle(
    ax=15.0, ay=20.0, az=0.0,
    bx=25.0, by=20.0, bz=0.0,
    height=2.5, thickness=0.3
)
```

### 精度分析

对于num_bins = 640，world范围[0, 30]：

```
网格大小 = 30.0 / 640 ≈ 0.047 米 ≈ 4.7 厘米

量化误差 ≤ 网格大小 / 2 ≈ 2.3 厘米
```

这个精度对于建筑尺度的管道路径规划是足够的。

## 使用示例

### 示例1：加载和解析布局

```python
from layout.layout import Layout

# 从文件加载
with open("scene_001.txt", "r") as f:
    content = f.read()

layout = Layout(content)

print(f"障碍物数量: {len(layout.obstacles)}")
print(f"端点数量: {len(layout.terminals)}")
print(f"路径节点数量: {len(layout.route_nodes)}")
```

### 示例2：归一化和反归一化

```python
# 归一化
original_layout = Layout(content)
original_ax = original_layout.obstacles[0].ax

layout.normalize_and_discretize(num_bins=640)
discretized_ax = layout.obstacles[0].ax

# 反归一化
layout.undiscretize_and_unnormalize(num_bins=640)
restored_ax = layout.obstacles[0].ax

print(f"原始: {original_ax}")
print(f"离散化: {discretized_ax}")
print(f"恢复: {restored_ax}")
print(f"误差: {abs(original_ax - restored_ax)}")
```

### 示例3：坐标变换

```python
# 平移
layout.translate([1.0, 2.0, 0.0])

# 旋转45度
import math
layout.rotate(math.pi / 4)

# 缩放2倍
layout.scale(2.0)

# 保存变换后的布局
output = layout.to_language_string()
with open("transformed.txt", "w") as f:
    f.write(output)
```

### 示例4：创建新布局

```python
from layout.entity import Obstacle, Terminal, RouteNode
from layout.layout import Layout

# 创建实体
obs1 = Obstacle(0, 0, 5, 0, 10, 5, 0, 2.5, 0.3)
obs2 = Obstacle(1, 5, 0, 0, 5, 10, 0, 2.5, 0.3)

term1 = Terminal(0, 1, 1, 1, 0, 0, 1, 0.15)
term2 = Terminal(1, 9, 9, 1, 0, 0, -1, 0.15)

nodes = [
    RouteNode(0, 1, 1, 1, 0),
    RouteNode(1, 3, 3, 1, -1),
    RouteNode(2, 6, 6, 1, -1),
    RouteNode(3, 9, 9, 1, 1)
]

# 创建布局
layout = Layout()
layout.obstacles = [obs1, obs2]
layout.terminals = [term1, term2]
layout.route_nodes = nodes

# 保存
output = layout.to_language_string()
print(output)
```

## 常见问题

### Q: 为什么法向量也需要归一化？

A: 法向量虽然是单位向量（模为1），但其分量范围是[-1, 1]。离散化时需要映射到[0, num_bins-1]，所以要归一化到[0, 1]。

### Q: RouteNode的terminal_id有什么用？

A: 标识该节点是否关联端点。通常第一个和最后一个节点关联端点（值为0或1），中间节点为-1。在路径规划时可以利用这个信息。

### Q: 离散化会损失精度吗？

A: 会有量化误差，但对于640 bins和30m场景，误差在2-3cm，对管道路径规划影响很小。可以通过增加bins数量提高精度。

### Q: 如何选择合适的num_bins？

A: 考虑因素：
- 场景尺度：更大的场景需要更多bins
- 精度要求：高精度任务需要更多bins  
- 计算资源：更多bins意味着更长的序列
- 建议：640-1024 bins对大多数场景足够

### Q: 旋转和平移的顺序重要吗？

A: 重要！变换不满足交换律。通常先旋转再平移：
```python
layout.rotate(angle)
layout.translate(t)
```

## 扩展阅读

- [SceneScript论文](https://arxiv.org/abs/2203.13064)
- [Shapely几何操作](https://shapely.readthedocs.io/)
- [scipy.spatial旋转](https://docs.scipy.org/doc/scipy/reference/spatial.transform.html)

---

本文档持续更新中。如有疑问或建议，请提交Issue。
