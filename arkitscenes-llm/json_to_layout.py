import json
import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
from collections import defaultdict, deque
import math

# 确保您的环境中有 spatiallm 包
# 如果没有，您需要将 Entity 和 Layout 类的定义复制到这里
from spatiallm.layout.entity import Obstacle, Terminal, RouteNode
from spatiallm.layout.layout import Layout

def get_dist(p1, p2):
    """计算两点欧几里得距离"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def simplify_path(points, tolerance=0.99):
    """
    移除路径中直线上的中间点，只保留起止点和拐点。
    
    Args:
        points: 点的列表 [(x,y,z), ...]
        tolerance: 判断共线的阈值。点积 > 0.99 视为共线。
    """
    if len(points) < 3:
        return points

    simplified = [points[0]]  # 总是保留起点
    
    # 将所有点预转换为 numpy 数组以加速计算
    np_points = [np.array(p) for p in points]
    
    for i in range(1, len(points) - 1):
        prev_p = np_points[i-1]
        curr_p = np_points[i]
        next_p = np_points[i+1]
        
        # 计算前后两个向量
        vec1 = curr_p - prev_p
        vec2 = next_p - curr_p
        
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)
        
        # 忽略极短的线段（重合点）
        if len1 < 1e-6 or len2 < 1e-6:
            continue
            
        # 归一化向量
        vec1_u = vec1 / len1
        vec2_u = vec2 / len2
        
        # 计算点积：1.0 表示方向完全相同（共线）
        # 如果点积小于阈值，说明方向发生了改变，是拐点
        if np.dot(vec1_u, vec2_u) < tolerance:
            simplified.append(points[i])
            
    simplified.append(points[-1]) # 总是保留终点
    return simplified

def process_json(json_path, output_txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    layout = Layout()
    
    # -----------------------------------------------------------------
    # 1. Obstacles (障碍物)
    # -----------------------------------------------------------------
    for idx, obs_data in enumerate(data.get("obstacles", [])):
        xmin, xmax, ymin, ymax, zmin, zmax = obs_data
        
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        width_x = xmax - xmin
        width_y = ymax - ymin
        
        # 简单的方向判定逻辑
        if width_x > width_y:
            ax, ay, az = xmin, center_y, zmin
            bx, by, bz = xmax, center_y, zmin
            thickness = width_y
        else:
            ax, ay, az = center_x, ymin, zmin
            bx, by, bz = center_x, ymax, zmin
            thickness = width_x
            
        height = zmax - zmin
        
        obstacle = Obstacle(
            id=idx,
            ax=ax, ay=ay, az=az,
            bx=bx, by=by, bz=bz,
            height=height, thickness=thickness
        )
        layout.obstacles.append(obstacle)
        
    # -----------------------------------------------------------------
    # 2. Terminals (管口)
    # -----------------------------------------------------------------
    terminals_data = data.get("terminals", [])
    layout_terminals = [] # 暂存 Python 对象用于后续距离计算
    
    for idx, term in enumerate(terminals_data):
        pos = term[0]
        normal = term[1]
        
        min_dist = float('inf')
        closest_obs_id = -1
        p_vec = np.array(pos)
        
        for obs in layout.obstacles:
            # 计算障碍物近似中心
            obs_center = np.array([
                (obs.ax + obs.bx) / 2, 
                (obs.ay + obs.by) / 2, 
                obs.az + obs.height / 2 
            ])

            dist = np.linalg.norm(p_vec - obs_center)
            if dist < min_dist:
                min_dist = dist
                closest_obs_id = obs.id
                
        t_obj = Terminal(
            id=idx,
            obstacle_id=closest_obs_id,
            position_x=pos[0], position_y=pos[1], position_z=pos[2],
            normal_x=normal[0], normal_y=normal[1], normal_z=normal[2],
            radius=0.2
        )
        layout.terminals.append(t_obj)
        layout_terminals.append(t_obj)
        
    # -----------------------------------------------------------------
    # 3. Route -> RouteNodes (图搜索 + 路径简化)
    # -----------------------------------------------------------------
    route_data = data.get("route", [])
    
    # 只有当存在路径数据且至少有两个端点时才处理
    if route_data and len(layout_terminals) >= 2:
        # A. 构建邻接表图 (Adjacency Graph)
        adj = defaultdict(list)
        def to_tuple(p): return tuple(p)
        all_points = set()
        
        for seg in route_data:
            u = to_tuple(seg[0])
            v = to_tuple(seg[1])
            if u == v: continue # 忽略长度为0的线段
            adj[u].append(v)
            adj[v].append(u)
            all_points.add(u)
            all_points.add(v)
            
        # B. 找到离 Terminal 0 和 Terminal 1 最近的图节点
        t0_pos = (layout_terminals[0].position_x, layout_terminals[0].position_y, layout_terminals[0].position_z)
        t1_pos = (layout_terminals[1].position_x, layout_terminals[1].position_y, layout_terminals[1].position_z)
        
        if not all_points:
             start_node = None
             end_node = None
        else:
             start_node = min(all_points, key=lambda p: get_dist(p, t0_pos))
             end_node = min(all_points, key=lambda p: get_dist(p, t1_pos))
        
        # C. BFS 搜索最短路径 (整理线段顺序)
        found_path = None
        
        if start_node and end_node:
            queue = deque([[start_node]])
            visited = {start_node}
            
            while queue:
                path = queue.popleft()
                curr = path[-1]
                
                if curr == end_node:
                    found_path = path
                    break
                
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_path = list(path)
                        new_path.append(neighbor)
                        queue.append(new_path)
        
        final_points = found_path if found_path else []
        
        # 调试：如果没找到路径，说明断开了
        if not final_points and len(all_points) > 0:
            # print(f"Warning: No connected path found in {os.path.basename(json_path)}.")
            pass 

        # D. 【新增】简化路径：去除共线点
        if final_points:
            final_points = simplify_path(final_points)

        # E. 生成 RouteNodes
        for idx, pt in enumerate(final_points):
            # 检查点是否与 Terminal 重合
            tid = -1
            for t in layout_terminals:
                t_vec = (t.position_x, t.position_y, t.position_z)
                if get_dist(pt, t_vec) < 0.2: 
                    tid = t.id
                    break
            
            node = RouteNode(
                id=idx,
                x=pt[0], y=pt[1], z=pt[2],
                terminal_id=tid
            )
            layout.route_nodes.append(node)
            
    # 4. 保存为 TXT
    with open(output_txt_path, 'w') as f:
        f.write(layout.to_language_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process JSON scenes to Layout TXT")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/data/storage2/liwentian/SpatialLM_pipe_gen/arkitscenes-spatiallm/scenes/meta", 
        help="Input directory containing JSON files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="/data/storage2/liwentian/SpatialLM_pipe_gen/arkitscenes-spatiallm/layout", 
        help="Output directory for TXT files"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
        
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files in {args.input_dir}")
    
    for json_file in tqdm(json_files, desc="Processing"):
        file_name = os.path.basename(json_file)
        file_base = os.path.splitext(file_name)[0]
        output_file = os.path.join(args.output_dir, f"{file_base}.txt")
        
        try:
            process_json(json_file, output_file)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")