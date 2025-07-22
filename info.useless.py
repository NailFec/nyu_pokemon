import numpy as np
import json

# 加载数据
print("Loading data...")
train_data = np.load('pokemon/pokemon_train/train_data.npy')
train_labels = np.load('pokemon/pokemon_train/train_labels.npy')

# 读取类别映射
with open('pokemon/types2label.txt', 'r') as f:
    types2label = eval(f.read())  # 或者用 json.loads 如果格式严格

print("=== 数据基本信息 ===")
print(f"Data shape: {train_data.shape}")
print(f"Labels shape: {train_labels.shape}")
print(f"Data dtype: {train_data.dtype}")
print(f"Labels dtype: {train_labels.dtype}")
print(f"Number of classes: {len(types2label)}")

print("\n=== 数据值范围 ===")
print(f"Data min: {train_data.min()}")
print(f"Data max: {train_data.max()}")
print(f"Data mean: {train_data.mean():.3f}")
print(f"Data std: {train_data.std():.3f}")

print("\n=== 标签信息 ===")
print(f"Labels min: {train_labels.min()}")
print(f"Labels max: {train_labels.max()}")
print(f"Unique labels count: {len(np.unique(train_labels))}")

print("\n=== 类别分布（前20个） ===")
unique_labels, counts = np.unique(train_labels, return_counts=True)
for i in range(min(20, len(unique_labels))):
    print(f"Class {unique_labels[i]}: {counts[i]} samples")

print("\n=== 类别映射示例（前10个） ===")
label2types = {v: k for k, v in types2label.items()}
for i in range(min(10, len(label2types))):
    if i in label2types:
        print(f"Label {i}: {label2types[i]}")

print("\n=== 检查数据是否需要归一化 ===")
print("前3个样本的前几个像素值:")
for i in range(3):
    print(f"Sample {i}: {train_data[i, 0, 0, :5]}")

print("\n=== Pokemon类型分析 ===")
single_type_count = 0
dual_type_count = 0

for type_combo in types2label.keys():
    if ',' in type_combo:
        dual_type_count += 1
    else:
        single_type_count += 1

print(f"Single type combinations: {single_type_count}")
print(f"Dual type combinations: {dual_type_count}")
print(f"Total combinations: {len(types2label)}")

# 分析所有可能的单独类型
all_types = set()
for type_combo in types2label.keys():
    types_in_combo = type_combo.split(',')
    for t in types_in_combo:
        all_types.add(t.strip())

print(f"\n=== 所有Pokemon类型 ===")
print(f"Total unique types: {len(all_types)}")
print(f"Types: {sorted(list(all_types))}")