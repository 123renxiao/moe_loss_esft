import os
import glob
import pandas as pd
import random

# 配置路径
# 使用脚本所在目录作为数据根目录，这样无论在哪里运行脚本都能找到正确路径
data_root = os.path.dirname(os.path.abspath(__file__))
pcd_dir = os.path.join(data_root, "pcd")
print(f"PCD 目录: {pcd_dir}")
layout_dir = os.path.join(data_root, "layout")
output_csv = os.path.join(data_root, "split.csv")

def main():
    # 1. 获取所有文件名
    print(f"正在扫描 {pcd_dir} ...")
    pcd_files = glob.glob(os.path.join(pcd_dir, "*.ply"))
    print(f"正在扫描 {pcd_dir} ...")
    print(f"正在扫描 {layout_dir} ...")
    layout_files = glob.glob(os.path.join(layout_dir, "*.txt"))

    # 2. 提取文件 ID (去掉扩展名)
    pcd_ids = set(os.path.basename(f).split(".")[0] for f in pcd_files)
    layout_ids = set(os.path.basename(f).split(".")[0] for f in layout_files)

    # 3. 找出交集 (即同时拥有 pcd 和 layout 的数据)
    valid_ids = list(pcd_ids & layout_ids)
    valid_ids.sort()

    print("-" * 30)
    print(f"PCD 文件总数: {len(pcd_ids)}")
    print(f"Layout 文件总数: {len(layout_ids)}")
    print(f"配对成功 (有效数据): {len(valid_ids)}")
    print("-" * 30)

    if not valid_ids:
        print("错误: 没有找到配对的数据！请检查文件名是否一致。")
        return

    # 4. 随机划分训练集和验证集 (例如 90% 训练, 10% 验证)
    random.seed(42) # 固定种子保证结果可复现
    data = []
    for scene_id in valid_ids:
        split = "train" if random.random() < 0.9 else "val"
        data.append({"id": scene_id, "split": split})

    # 5. 保存为 CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"成功生成划分文件: {output_csv}")
    print("现在您可以运行 create_dataset.py 了。")

if __name__ == "__main__":
    main()