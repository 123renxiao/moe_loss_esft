from typing import Dict

# 用于存储当前 Batch 的 Tag (Tensor)
CURRENT_BATCH_TAGS = None

# 用于存储当前 Batch 的 Attention Mask (Tensor)
CURRENT_ATTENTION_MASK = None

# 用于存储路由统计信息
# 结构: {layer_idx: {tag_id: {expert_id: count}}}
ROUTER_STATS: Dict[int, Dict[int, Dict[int, int]]] = {}
