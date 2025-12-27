import json
import os
import argparse
import numpy as np
import glob
from tqdm import tqdm
from spatiallm.layout.entity import Obstacle, Terminal, RouteNode
from spatiallm.layout.layout import Layout

def process_json(json_path, output_txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    layout = Layout()
    
    # 1. Obstacles -> Obstacles
    for idx, obs_data in enumerate(data.get("obstacles", [])):
        xmin, xmax, ymin, ymax, zmin, zmax = obs_data
        
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        width_x = xmax - xmin
        width_y = ymax - ymin
        
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
        
    # 2. Terminals -> Terminals
    terminals_data = data.get("terminals", [])
    for idx, term in enumerate(terminals_data):
        pos = term[0]
        normal = term[1]
        
        # Removed obstacle binding logic
                
        t_obj = Terminal(
            id=idx,
            position_x=pos[0], position_y=pos[1], position_z=pos[2],
            normal_x=normal[0], normal_y=normal[1], normal_z=normal[2],
            radius=0.2
        )
        layout.terminals.append(t_obj)
        
    # 3. Route -> RouteNodes
    route_data = data.get("route", [])
    if route_data:
        points = [route_data[0][0]] 
        for seg in route_data:
            points.append(seg[1]) 
            
        for idx, pt in enumerate(points):
            tid = -1
            pt_vec = np.array(pt)
            for t in layout.terminals:
                t_vec = np.array([t.position_x, t.position_y, t.position_z])
                if np.linalg.norm(pt_vec - t_vec) < 0.1: 
                    tid = t.id
                    break
            
            node = RouteNode(
                id=idx,
                x=pt[0], y=pt[1], z=pt[2],
                terminal_id=tid
            )
            layout.route_nodes.append(node)
            
    # 4. Save
    with open(output_txt_path, 'w') as f:
        f.write(layout.to_language_string())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process JSON scenes to Layout TXT")
    parser.add_argument("--input_dir", type=str, default="/data/storage2/liwentian/SpatialLM_pipe_gen/arkitscenes-spatiallm/scenes", help="Input directory containing JSON files")
    parser.add_argument("--output_dir", type=str, default="/data/storage2/liwentian/SpatialLM_pipe_gen/arkitscenes-spatiallm/layout", help="Output directory for TXT files")
    
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
