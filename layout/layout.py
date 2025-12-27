import numpy as np
from scipy.spatial.transform import Rotation as R
from .entity import Obstacle, Terminal, RouteNode, NORMALIZATION_PRESET

class Layout:
    def __init__(self, s: str = None):
        self.obstacles = []   
        self.terminals = []   
        self.route_nodes = [] 

        if s:
            self.from_str(s)

    @staticmethod
    def get_grid_size(num_bins):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        return (world_max - world_min) / num_bins

    def get_entities(self):
        # 【修改】确保返回所有类型的实体，用于批量变换
        return self.obstacles + self.terminals + self.route_nodes

    def from_str(self, s: str):
        s = s.lstrip("\n")
        lines = s.split("\n")
        
        # 维护一个 obstacle id 集合，用于校验 terminal 的引用
        existing_obstacles = set()

        for line in lines:
            try:
                if "=" not in line or "(" not in line: continue
                
                label_part, value_part = line.split("=")
                if "_" not in label_part: continue
                
                entity_label, entity_id_str = label_part.split("_")
                entity_id = int(entity_id_str)
                
                start_pos = value_part.find("(")
                end_pos = value_part.find(")")
                params_str = value_part[start_pos+1:end_pos]
                params = params_str.split(",")

                if entity_label == "obstacle":  # 【修改】解析 obstacle
                    if len(params) < 8: continue
                    args = dict(zip(
                        ["ax", "ay", "az", "bx", "by", "bz", "height", "thickness"], 
                        params[:8]
                    ))
                    self.obstacles.append(Obstacle(id=entity_id, **args))
                    existing_obstacles.add(entity_id)
                    
                elif entity_label == "terminal":
                    if len(params) < 7: continue
                    
                    args = dict(zip(
                        ["position_x", "position_y", "position_z", 
                         "normal_x", "normal_y", "normal_z", "radius"],
                        params[:7]
                    ))
                    self.terminals.append(Terminal(id=entity_id, **args))
                    
                elif entity_label == "node":
                    if len(params) < 4: continue
                    args = dict(zip(
                        ["x", "y", "z", "terminal_id"],
                        params[:4]
                    ))
                    self.route_nodes.append(RouteNode(id=entity_id, **args))
                    
            except Exception as e:
                # print(f"Parse error on line '{line}': {e}")
                continue

    def to_boxes(self):
        boxes = []
        
        # 1. Obstacles -> Boxes
        for obs in self.obstacles:
            start = np.array([obs.ax, obs.ay, obs.az])
            end = np.array([obs.bx, obs.by, obs.bz])
            diff = end - start
            length = np.linalg.norm(diff)
            angle = np.arctan2(diff[1], diff[0])
            
            center = (start + end) / 2
            center[2] += obs.height / 2 
            
            rot_mat = R.from_rotvec([0, 0, angle]).as_matrix()
            
            boxes.append({
                "id": obs.id,
                "class": "Obstacle",
                "label": f"Obs_{obs.id}",
                "center": center,
                "rotation": rot_mat,
                "scale": np.array([length, obs.thickness, obs.height])
            })

        # 2. Terminals -> Boxes
        for term in self.terminals:
            center = np.array([term.position_x, term.position_y, term.position_z])
            r = term.radius
            boxes.append({
                "id": 1000 + term.id,
                "class": "Terminal",
                "label": f"Term_{term.id}",
                "center": center,
                "rotation": np.eye(3), # Terminal 显示为球/方块，旋转不明显
                "scale": np.array([r*2, r*2, r*2])
            })

        # 3. RouteNodes -> Boxes (Visualizing Segments)
        sorted_nodes = sorted(self.route_nodes, key=lambda n: n.id)
        for i in range(len(sorted_nodes) - 1):
            n1 = sorted_nodes[i]
            n2 = sorted_nodes[i+1]
            
            p1 = np.array([n1.x, n1.y, n1.z])
            p2 = np.array([n2.x, n2.y, n2.z])
            diff = p2 - p1
            length = np.linalg.norm(diff)
            if length < 1e-6: continue
            
            center = (p1 + p2) / 2
            
            # 计算 3D 旋转矩阵，将X轴对齐到向量 diff
            x_axis = np.array([1, 0, 0])
            target_vec = diff / length
            
            # 计算旋转轴和角度
            rot_axis = np.cross(x_axis, target_vec)
            sin_angle = np.linalg.norm(rot_axis)
            cos_angle = np.dot(x_axis, target_vec)
            
            if sin_angle < 1e-6:
                # 平行或反向
                rot_mat = np.eye(3) if cos_angle > 0 else -np.eye(3)
            else:
                rot_axis = rot_axis / sin_angle
                angle = np.arctan2(sin_angle, cos_angle)
                rot_mat = R.from_rotvec(rot_axis * angle).as_matrix()

            boxes.append({
                "id": 2000 + n1.id,
                "class": "Route",
                "label": f"Seg_{n1.id}",
                "center": center,
                "rotation": rot_mat,
                "scale": np.array([length, 0.1, 0.1])
            })

        return boxes

    # 批量变换
    def normalize_and_discretize(self, num_bins):
        for e in self.get_entities(): e.normalize_and_discretize(num_bins)

    def undiscretize_and_unnormalize(self, num_bins):
        for e in self.get_entities(): e.undiscretize_and_unnormalize(num_bins)

    def translate(self, t):
        for e in self.get_entities(): e.translate(t)

    def rotate(self, a):
        for e in self.get_entities(): e.rotate(a)

    def scale(self, s):
        for e in self.get_entities(): e.scale(s)

    def reorder_entities(self):
        # 1. 障碍物按空间排序
        if self.obstacles:
            self.obstacles.sort(key=lambda o: (o.ax, o.ay))
            # 重建索引映射表
            old_to_new_ids = {}
            for i, obs in enumerate(self.obstacles):
                old_to_new_ids[obs.id] = i
                obs.id = i
            
        # 2. Terminals 按空间排序
        if self.terminals:
            self.terminals.sort(key=lambda t: (t.position_x, t.position_y))
            for i, t in enumerate(self.terminals): t.id = i
            
        # 3. RouteNodes 严格按 ID 排序
        if self.route_nodes:
            self.route_nodes.sort(key=lambda n: n.id)
            for i, n in enumerate(self.route_nodes): n.id = i

    def to_language_string(self):
        lines = []
        for o in self.obstacles: lines.append(o.to_language_string())
        for t in self.terminals: lines.append(t.to_language_string())
        for n in self.route_nodes: lines.append(n.to_language_string())
        return "\n".join(lines)
