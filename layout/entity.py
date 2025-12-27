# Copyright (c) Manycore Tech Inc. and affiliates.
# All rights reserved.

"""
This code is derived from the SceneScript language sequence and entity parameters.

Reference: https://github.com/facebookresearch/scenescript/blob/main/src/data/language_sequence.py
"""

# terminal_0=Terminal(px, py, pz, nx, ny, nz, radius)

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

NORMALIZATION_PRESET = {
    "world": (0.0, 30.0),
    "height": (0.0, 10),
    "width": (0.0, 25.6),
    "scale": (0.0, 0.5),
    "angle": (-6.2832, 6.2832),
}


@dataclass
class Obstacle:
    id: int
    ax: float
    ay: float
    az: float
    bx: float
    by: float
    bz: float
    height: float
    thickness: float
    entity_label: str = "obstacle"

    def __post_init__(self):
        self.id = int(self.id)
        self.ax = float(self.ax)
        self.ay = float(self.ay)
        self.az = float(self.az)
        self.bx = float(self.bx)
        self.by = float(self.by)
        self.bz = float(self.bz)
        self.height = float(self.height)
        self.thickness = float(self.thickness)

    def rotate(self, angle: float):
        obstacle_start = np.array([self.ax, self.ay, self.az])
        obstacle_end = np.array([self.bx, self.by, self.bz])
        rotmat = R.from_rotvec([0, 0, angle]).as_matrix()
        obstacle_start = rotmat @ obstacle_start
        obstacle_end = rotmat @ obstacle_end

        self.ax = obstacle_start[0]
        self.ay = obstacle_start[1]
        self.az = obstacle_start[2]
        self.bx = obstacle_end[0]
        self.by = obstacle_end[1]
        self.bz = obstacle_end[2]

    def translate(self, translation: np.ndarray):
        self.ax += translation[0]
        self.ay += translation[1]
        self.az += translation[2]
        self.bx += translation[0]
        self.by += translation[1]
        self.bz += translation[2]

    def scale(self, scaling: float):
        self.height *= scaling
        self.thickness *= scaling
        self.ax *= scaling
        self.ay *= scaling
        self.az *= scaling
        self.bx *= scaling
        self.by *= scaling
        self.bz *= scaling

    def normalize_and_discretize(self, num_bins):
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]

        self.height = (self.height - height_min) / (height_max - height_min) * num_bins
        self.thickness = (
            (self.thickness - height_min) / (height_max - height_min) * num_bins
        )
        self.ax = (self.ax - world_min) / (world_max - world_min) * num_bins
        self.ay = (self.ay - world_min) / (world_max - world_min) * num_bins
        self.az = (self.az - world_min) / (world_max - world_min) * num_bins
        self.bx = (self.bx - world_min) / (world_max - world_min) * num_bins
        self.by = (self.by - world_min) / (world_max - world_min) * num_bins
        self.bz = (self.bz - world_min) / (world_max - world_min) * num_bins

        self.height = np.clip(int(self.height), 0, num_bins - 1)
        self.thickness = np.clip(int(self.thickness), 0, num_bins - 1)
        self.ax = np.clip(int(self.ax), 0, num_bins - 1)
        self.ay = np.clip(int(self.ay), 0, num_bins - 1)
        self.az = np.clip(int(self.az), 0, num_bins - 1)
        self.bx = np.clip(int(self.bx), 0, num_bins - 1)
        self.by = np.clip(int(self.by), 0, num_bins - 1)
        self.bz = np.clip(int(self.bz), 0, num_bins - 1)

    def undiscretize_and_unnormalize(self, num_bins):
        height_min, height_max = NORMALIZATION_PRESET["height"]
        world_min, world_max = NORMALIZATION_PRESET["world"]

        # undiscretize
        self.height = self.height / num_bins
        self.thickness = self.thickness / num_bins
        self.ax = self.ax / num_bins
        self.ay = self.ay / num_bins
        self.az = self.az / num_bins
        self.bx = self.bx / num_bins
        self.by = self.by / num_bins
        self.bz = self.bz / num_bins

        # unnormalize
        self.height = self.height * (height_max - height_min) + height_min
        self.thickness = self.thickness * (height_max - height_min) + height_min
        self.ax = self.ax * (world_max - world_min) + world_min
        self.ay = self.ay * (world_max - world_min) + world_min
        self.az = self.az * (world_max - world_min) + world_min
        self.bx = self.bx * (world_max - world_min) + world_min
        self.by = self.by * (world_max - world_min) + world_min
        self.bz = self.bz * (world_max - world_min) + world_min

    def to_language_string(self):
        capitalized_label = self.entity_label.capitalize()
        # obstacle_0=Obstacle(a_x,a_y,a_z,b_x,b_y,b_z,height,thickness)
        language_string = f"{self.entity_label}_{self.id}={capitalized_label}({self.ax},{self.ay},{self.az},{self.bx},{self.by},{self.bz},{self.height},{self.thickness})"
        return language_string

    def sort_key(self):
        # Lex-sort corners
        obstacle_start = np.array([self.ax, self.ay, self.az])
        obstacle_end = np.array([self.bx, self.by, self.bz])
        corners = np.stack([obstacle_start, obstacle_end])  # [2, 3]

        idx = np.lexsort(corners.T)  # [2]. Sorts by z, y, x.
        corner_1_ordered, corner_2_ordered = corners[idx]

        # Sort obstacle-corners
        self.ax, self.ay, self.az = corner_1_ordered
        self.bx, self.by, self.bz = corner_2_ordered

        return np.concatenate([corner_2_ordered, corner_1_ordered])


@dataclass
class Terminal:
    id: int
    position_x: float
    position_y: float
    position_z: float
    normal_x: float
    normal_y: float
    normal_z: float
    radius: float
    entity_label: str = "terminal"

    def __post_init__(self):
        self.id = int(self.id)
        self.position_x = float(self.position_x)
        self.position_y = float(self.position_y)
        self.position_z = float(self.position_z)
        self.normal_x = float(self.normal_x)
        self.normal_y = float(self.normal_y)
        self.normal_z = float(self.normal_z)
        self.radius = float(self.radius)

    def rotate(self, angle: float):
        # 1. 旋转位置 (Position)
        center = np.array([self.position_x, self.position_y, self.position_z])
        rotmat = R.from_rotvec([0, 0, angle]).as_matrix()
        new_center = rotmat @ center
        
        self.position_x = new_center[0]
        self.position_y = new_center[1]
        self.position_z = new_center[2]

        # 2. 旋转法向量 (Normal) - 关键差异点
        normal = np.array([self.normal_x, self.normal_y, self.normal_z])
        new_normal = rotmat @ normal
        
        self.normal_x = new_normal[0]
        self.normal_y = new_normal[1]
        self.normal_z = new_normal[2]

    def translate(self, translation: np.ndarray):
        # 仅平移位置，方向(Normal)不受平移影响
        self.position_x += translation[0]
        self.position_y += translation[1]
        self.position_z += translation[2]

    def scale(self, scaling: float):
        # 缩放位置和半径，方向(Normal)是单位向量，通常不缩放
        self.position_x *= scaling
        self.position_y *= scaling
        self.position_z *= scaling
        self.radius *= scaling

    def normalize_and_discretize(self, num_bins):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_min, scale_max = NORMALIZATION_PRESET["scale"] 
        norm_min, norm_max = -1.0, 1.0

        self.radius = (self.radius - scale_min) / (scale_max - scale_min) * num_bins

        self.position_x = (self.position_x - world_min) / (world_max - world_min) * num_bins
        self.position_y = (self.position_y - world_min) / (world_max - world_min) * num_bins
        self.position_z = (self.position_z - world_min) / (world_max - world_min) * num_bins

        self.normal_x = (self.normal_x - norm_min) / (norm_max - norm_min) * num_bins
        self.normal_y = (self.normal_y - norm_min) / (norm_max - norm_min) * num_bins
        self.normal_z = (self.normal_z - norm_min) / (norm_max - norm_min) * num_bins

        self.radius = np.clip(int(self.radius), 0, num_bins - 1)
        
        self.position_x = np.clip(int(self.position_x), 0, num_bins - 1)
        self.position_y = np.clip(int(self.position_y), 0, num_bins - 1)
        self.position_z = np.clip(int(self.position_z), 0, num_bins - 1)
        
        self.normal_x = np.clip(int(self.normal_x), 0, num_bins - 1)
        self.normal_y = np.clip(int(self.normal_y), 0, num_bins - 1)
        self.normal_z = np.clip(int(self.normal_z), 0, num_bins - 1)

    def undiscretize_and_unnormalize(self, num_bins):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        scale_min, scale_max = NORMALIZATION_PRESET["scale"]
        norm_min, norm_max = -1.0, 1.0

        self.radius = self.radius / num_bins
        
        self.position_x = self.position_x / num_bins
        self.position_y = self.position_y / num_bins
        self.position_z = self.position_z / num_bins
        
        self.normal_x = self.normal_x / num_bins
        self.normal_y = self.normal_y / num_bins
        self.normal_z = self.normal_z / num_bins

        self.radius = self.radius * (scale_max - scale_min) + scale_min
        
        self.position_x = self.position_x * (world_max - world_min) + world_min
        self.position_y = self.position_y * (world_max - world_min) + world_min
        self.position_z = self.position_z * (world_max - world_min) + world_min
        
        self.normal_x = self.normal_x * (norm_max - norm_min) + norm_min
        self.normal_y = self.normal_y * (norm_max - norm_min) + norm_min
        self.normal_z = self.normal_z * (norm_max - norm_min) + norm_min

    def to_language_string(self):
        capitalized_label = self.entity_label.capitalize()
        # id 取模逻辑与 Door 保持一致 (防止 ID 过大)
        self.id = self.id % 1000
        
        # 格式: terminal_0=Terminal(px, py, pz, nx, ny, nz, radius)
        language_string = (
            f"{self.entity_label}_{self.id}={capitalized_label}("
            f"{self.position_x},{self.position_y},{self.position_z},"
            f"{self.normal_x},{self.normal_y},{self.normal_z},"
            f"{self.radius})"
        )
        return language_string

    def sort_key(self):
        return np.array([self.position_x, self.position_y])
    
@dataclass
class RouteNode:
    id: int
    x: float
    y: float
    z: float
    terminal_id: int = -1 # 默认为 -1
    entity_label: str = "node"

    def __post_init__(self):
        self.id = int(self.id)
        self.x = float(self.x)
        self.y = float(self.y)
        self.z = float(self.z)
        self.terminal_id = int(self.terminal_id)

    def rotate(self, angle: float):
        # 仅旋转坐标
        pt = np.array([self.x, self.y, self.z])
        rotmat = R.from_rotvec([0, 0, angle]).as_matrix()
        pt = rotmat @ pt
        self.x, self.y, self.z = pt

    def translate(self, translation: np.ndarray):
        self.x += translation[0]
        self.y += translation[1]
        self.z += translation[2]

    def scale(self, scaling: float):
        self.x *= scaling
        self.y *= scaling
        self.z *= scaling

    def normalize_and_discretize(self, num_bins):
        # 使用 world 范围
        world_min, world_max = NORMALIZATION_PRESET["world"]
        
        self.x = (self.x - world_min) / (world_max - world_min) * num_bins
        self.y = (self.y - world_min) / (world_max - world_min) * num_bins
        self.z = (self.z - world_min) / (world_max - world_min) * num_bins
        
        self.x = np.clip(int(self.x), 0, num_bins - 1)
        self.y = np.clip(int(self.y), 0, num_bins - 1)
        self.z = np.clip(int(self.z), 0, num_bins - 1)

    def undiscretize_and_unnormalize(self, num_bins):
        world_min, world_max = NORMALIZATION_PRESET["world"]
        
        self.x = self.x / num_bins
        self.y = self.y / num_bins
        self.z = self.z / num_bins
        
        self.x = self.x * (world_max - world_min) + world_min
        self.y = self.y * (world_max - world_min) + world_min
        self.z = self.z * (world_max - world_min) + world_min

    def to_language_string(self):
        # ID 不取模，或者取一个较大的模，防止长路径重复
        # 格式: node_0=Node(x, y, z, terminal_id)
        # 注意 terminal_id 如果是 -1，在字符串中也需要体现
        return f"{self.entity_label}_{self.id}=Node({self.x},{self.y},{self.z},{self.terminal_id})"

    def sort_key(self):
        # 关键：按 ID 排序
        return np.array([self.id])
