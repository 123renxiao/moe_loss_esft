"""
管道路径碰撞检测评估脚本
==========================
本脚本评估生成的管道路径是否与场景中的障碍物发生碰撞。

碰撞检测算法：
1. Z轴快速排斥测试（粗筛选）
2. XY平面2D投影相交检测
3. 3D精确穿透检测
4. 端点容差处理（允许路径首尾插入障碍物一定深度）

评估指标：
- 通过率（Success Rate）
- 失败原因（具体哪段路径碰撞了哪个障碍物）

作者：Li Wentian
日期：2025年
"""

import os
import glob
import re
import math
from shapely.geometry import Polygon, LineString, Point

# =================配置区域=================
EPSILON = 1e-5          # 浮点数计算容差，用于处理浮点误差
TERMINAL_TOLERANCE = 0.1  # 端点容差：允许管路头尾插入障碍物的深度
                          # 这是因为端点通常需要穿墙或穿障碍物才能连接
RESULTS_DIR = '/data/storage2/liwentian/SpatialLM_pipe_gen/results_val_spatiallm_17'
OUTPUT_FILE = 'evaluation_spatiallm_report_17.txt'
# =========================================

class Obstacle:
    """
    障碍物类
    
    表示3D空间中的障碍物，由两个端点(A, B)和尺寸(height, thickness)定义。
    障碍物在XY平面的投影是一个矩形，Z方向有一定高度。
    """
    def __init__(self, id, ax, ay, az, bx, by, bz, height, thickness):
        self.id = int(id)
        self.ax = float(ax)
        self.ay = float(ay)
        self.az = float(az)
        self.bx = float(bx)
        self.by = float(by)
        self.bz = float(bz)
        self.height = float(height)
        self.thickness = float(thickness)
        self._polygon_cache = None  # 缓存XY平面投影多边形

    def get_xy_polygon(self):
        """
        获取障碍物在XY平面的投影多边形（矩形）
        
        算法：
        1. 计算AB方向向量
        2. 计算垂直方向（法向量）
        3. 沿法向量向两侧扩展thickness/2
        4. 得到4个角点，构成矩形
        
        返回:
            Shapely Polygon对象，如果障碍物退化为点则返回None
        """
        if self._polygon_cache is not None:
            return self._polygon_cache

        dx = self.bx - self.ax
        dy = self.by - self.ay
        length = math.sqrt(dx*dx + dy*dy)
        
        if length == 0: return None  # 退化为点

        # 单位方向向量
        ux = dx / length
        uy = dy / length
        # 法向量（垂直方向）
        nx, ny = -uy, ux
        
        # 沿法向量扩展thickness/2
        ox = nx * (self.thickness / 2.0)
        oy = ny * (self.thickness / 2.0)
        
        # 4个角点
        c1 = (self.ax + ox, self.ay + oy)
        c2 = (self.bx + ox, self.by + oy)
        c3 = (self.bx - ox, self.by - oy)
        c4 = (self.ax - ox, self.ay - oy)
        
        self._polygon_cache = Polygon([c1, c2, c3, c4])
        return self._polygon_cache

    def get_z_range(self):
        """
        获取障碍物的Z方向范围
        
        返回:
            (z_min, z_max) 元组
        """
        base_z = min(self.az, self.bz)
        return (base_z, base_z + self.height)

class Node:
    def __init__(self, id, x, y, z, terminal_id):
        self.id = int(id)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.terminal_id = int(terminal_id)

def parse_file(filepath):
    obstacles = []
    nodes = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            obs_match = re.match(r'obstacle_(\d+)=Obstacle\((.*)\)', line)
            if obs_match:
                oid = obs_match.group(1)
                params = [float(x) for x in obs_match.group(2).split(',')]
                if len(params) == 8: obstacles.append(Obstacle(oid, *params))
                continue

            node_match = re.match(r'node_(\d+)=Node\((.*)\)', line)
            if node_match:
                nid = node_match.group(1)
                params = [x for x in node_match.group(2).split(',')]
                x, y, z = float(params[0]), float(params[1]), float(params[2])
                tid = int(params[3])
                nodes.append(Node(nid, x, y, z, tid))
                continue
                
    nodes.sort(key=lambda n: n.id)
    return obstacles, nodes

def check_segment_obstacle_collision(p1, p2, obstacle):
    """
    检查线段是否与障碍物发生碰撞
    
    参数:
        p1: 线段起点 (x, y, z)
        p2: 线段终点 (x, y, z)
        obstacle: Obstacle对象
    
    返回:
        True表示发生碰撞，False表示无碰撞
    
    算法流程:
        1. Z轴快速排斥测试
        2. XY平面投影相交检测
        3. 对相交部分进行3D穿透验证
    """
    # 1. Z轴快速排斥测试
    # 如果线段和障碍物在Z方向没有重叠，则一定不碰撞
    seg_z_min = min(p1[2], p2[2])
    seg_z_max = max(p1[2], p2[2])
    obs_z_min, obs_z_max = obstacle.get_z_range()
    
    if seg_z_min > obs_z_max - EPSILON or seg_z_max < obs_z_min + EPSILON:
        return False
        
    # 2. 获取障碍物的XY平面投影多边形
    poly = obstacle.get_xy_polygon()
    if poly is None: return False

    xy_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    # === 垂直管路特殊处理 ===
    # 如果管路是垂直的（XY平面投影为点），需要特殊检测
    if xy_len < EPSILON:
        point_2d = Point(p1[0], p1[1])
        if poly.contains(point_2d) or poly.touches(point_2d):
            # 计算Z方向重叠长度
            overlap_min = max(seg_z_min, obs_z_min)
            overlap_max = min(seg_z_max, obs_z_max)
            # 只有实质性重叠才算碰撞（容差处理）
            if overlap_max - overlap_min > EPSILON:
                return True
        return False

    # === 普通管路（非垂直）===
    line_geom = LineString([(p1[0], p1[1]), (p2[0], p2[1])])
    if not line_geom.intersects(poly): return False
        
    intersection = line_geom.intersection(poly)
    if intersection.is_empty: return False

    # 3. 检查 3D 穿透
    # 对于相交部分的每一段，检查其在Z方向是否真的穿过障碍物
    def check_piece_collision(geom):
        """递归检查几何体碰撞"""
        if isinstance(geom, Point): 
            return False  # 点接触不算碰撞
        elif isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) < 2: return False
            
            # 获取相交线段的起点和终点
            p_start_2d = Point(coords[0])
            p_end_2d = Point(coords[-1])
            
            # 计算这两个点在原始线段上的位置参数t (0到1)
            dist_start = line_geom.project(p_start_2d)
            dist_end = line_geom.project(p_end_2d)
            
            t_start = dist_start / xy_len
            t_end = dist_end / xy_len
            
            # 根据t计算这两个点的Z坐标
            z_start = p1[2] + t_start * (p2[2] - p1[2])
            z_end = p1[2] + t_end * (p2[2] - p1[2])
            
            # 检查这段在Z方向是否与障碍物重叠
            piece_z_min = min(z_start, z_end)
            piece_z_max = max(z_start, z_end)
            
            if piece_z_min > obs_z_max - EPSILON: return False
            if piece_z_max < obs_z_min + EPSILON: return False
            return True  # Z方向有重叠，确实碰撞
            
        elif hasattr(geom, 'geoms'):
            # 处理MultiLineString等复合几何体
            for part in geom.geoms:
                if check_piece_collision(part): return True
            return False
        return False

    return check_piece_collision(intersection)

# --- 端点容差处理函数 ---
def trim_segment(p_start, p_end, trim_dist):
    """
    将线段起点沿着终点方向移动一定距离
    
    用途：处理端点容差，允许路径首尾插入障碍物一定深度
    
    参数:
        p_start: 起点坐标 (x, y, z)
        p_end: 终点坐标 (x, y, z)
        trim_dist: 需要移动的距离
    
    返回:
        新的起点坐标，如果线段长度小于trim_dist则返回None
    """
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    dz = p_end[2] - p_start[2]
    
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    
    if length <= trim_dist:
        return None  # 线段太短，完全在容差范围内，跳过检查
    
    # 计算新起点位置
    ratio = trim_dist / length
    new_x = p_start[0] + dx * ratio
    new_y = p_start[1] + dy * ratio
    new_z = p_start[2] + dz * ratio
    
    return (new_x, new_y, new_z)

def evaluate_scene(filepath):
    """
    评估单个场景文件的路径规划结果
    
    参数:
        filepath: 场景结果文件路径（包含obstacles和nodes）
    
    返回:
        (score, message) 元组
        - score: 1表示通过，0表示失败
        - message: 成功信息或失败原因
    
    评估逻辑:
        遍历所有路径线段，检查是否与任何障碍物碰撞
        对首尾线段应用端点容差
    """
    obstacles, nodes = parse_file(filepath)
    if len(nodes) < 2: return 1, "Not enough nodes"
    
    total_segments = len(nodes) - 1
    
    # 遍历所有路径线段
    for i in range(total_segments):
        n1 = nodes[i]
        n2 = nodes[i+1]
        p1 = (n1.x, n1.y, n1.z)
        p2 = (n2.x, n2.y, n2.z)
        
        # === 关键修改：对首尾线段进行缩进处理 ===
        # 这是因为端点通常需要插入墙体或障碍物才能连接
        
        # 1. 如果是第一段 (Node 0 -> Node 1)，起点允许插入障碍物
        if i == 0:
            p1 = trim_segment(p1, p2, TERMINAL_TOLERANCE)
            if p1 is None: continue  # 第一段很短，且都在容差内，跳过检查

        # 2. 如果是最后一段 (Node N-1 -> Node N)，终点允许插入障碍物
        if i == total_segments - 1:
            p2 = trim_segment(p2, p1, TERMINAL_TOLERANCE)  # 注意：p2向p1缩进
            if p2 is None: continue  # 最后一段很短，跳过检查
            
        # ==========================================
        
        # 检查该线段是否与任何障碍物碰撞
        for obs in obstacles:
            if check_segment_obstacle_collision(p1, p2, obs):
                reason = f"Collision: Node {n1.id}->{n2.id} hit Obstacle {obs.id}"
                return 0, reason
            
    return 1, "Success"

def main():
    if not os.path.exists(RESULTS_DIR):
        print(f"Directory {RESULTS_DIR} not found.")
        return

    files = glob.glob(os.path.join(RESULTS_DIR, 'scene_*.txt'))
    files.sort()
    
    if not files:
        print("No files found.")
        return
    
    with open(OUTPUT_FILE, 'w') as f_out:
        
        def log(message):
            print(message)
            f_out.write(message + '\n')

        passed_files = []
        failed_files = []
        
        log(f"{'File Name':<35} | {'Status':<10} | {'Info'}")
        log("-" * 85)
        
        for f in files:
            score, msg = evaluate_scene(f)
            filename = os.path.basename(f)
            
            if score == 1:
                status = "PASS"
                passed_files.append(filename)
                log(f"{filename:<35} | {status:<10} | {msg}")
            else:
                status = "FAIL"
                failed_files.append(filename)
                log(f"{filename:<35} | {status:<10} | {msg}")
                
        log("-" * 85)
        log(f"Total: {len(files)} | Passed: {len(passed_files)} | Failed: {len(failed_files)}")
        if len(files) > 0:
            log(f"Success Rate: {len(passed_files)/len(files)*100:.2f}%")
        
        if failed_files:
            log("\nList of Failed Files:")
            for fail in failed_files:
                log(f" - {fail}")

    print(f"\nResults have been saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()