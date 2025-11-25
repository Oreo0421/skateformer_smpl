import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

"""
将 all_transformed_joints.npz 转成 SkateFormer 需要的格式：

输出：
  ./skateformer_data/
      train_data_joint.npy   # (N_train, T, J, 3)
      val_data_joint.npy     # (N_val,   T, J, 3)
      train_label.pkl        # [int, int, ...]
      val_label.pkl

目前默认：
  1. 从 all_transformed_joints.npz 中取第一个数组作为 joints 数据
  2. 假设 joints shape 为 (N, T, J, C)，其中 C>=2
  3. 暂时用“假标签”：所有样本 label=0
     —— 只是为了验证“合成数据能否跑通训练”，
        之后你可以自己换成真实动作类别标签。
"""

NPZ_PATH = "all_transformed_joints.npz"
OUT_DIR = "skateformer_data"
MAX_FRAMES = 300   # SkateFormer 通常用 300 帧对齐，可按需要改

def load_joints_from_npz(npz_path):
    data = np.load(npz_path)
    print("Found keys in npz:", data.files)
    key = data.files[0]
    print("Use key:", key)
    arr = data[key]
    print("Original array shape:", arr.shape)

def load_joints_from_npz(npz_path):
    data = np.load(npz_path)
    print("Found keys in npz:", data.files)

    # 明确用 joints_3d 这个 key
    if "joints_3d" in data.files:
        key = "joints_3d"
    else:
        key = data.files[0]

    print("Use key:", key)
    arr = data[key]
    print("Original array shape:", arr.shape)


    if arr.ndim == 3:
        # 解释为 (T, J, C)，把它当成 1 个样本：
        # -> (1, T, J, C)
        arr = arr[np.newaxis, ...]
        print("Treat as (N=1, T, J, C):", arr.shape)

    if arr.ndim != 4:
        raise ValueError(f"Expect joints data with ndim=4, got shape {arr.shape}")

    return arr  # (N, T, J, C)


    if arr.ndim != 4:
        raise ValueError(f"Expect joints data with ndim=4, got shape {arr.shape}")

    return arr  # (N, T, J, C)

def normalize_and_pad(joints, max_frames=300):
    """
    joints: (N, T, J, C)，C>=3 (例如 XYZ)
    返回：  (N, max_frames, J, 3)
    1. 只保留前三维作为 (x, y, z)
    2. 以第 2 号关节 (index 1) 为中心平移
    3. pad / 截断到统一帧数 max_frames
    """
    N, T, J, C = joints.shape
    if C < 3:
        raise ValueError(f"Expect at least 3 coords per joint, got C={C}")

    print(f"Before normalize: N={N}, T={T}, J={J}, C={C}")

    # 只用前3维
    joints = joints[..., :3]  # (N, T, J, 3)

    # 平移：用关节 1 (也可以改为其他，你的骨架定义里“躯干中心”哪个就用哪个)
    origin = joints[:, 0, 1:2, :]  # (N, 1, 1, 3)，第 1 帧的第 2 个点
    joints_centered = joints - origin

    # 计算每个样本的尺度（例如所有关节的平均距离或某两点距离，这里用简单范数）
    # 为了稳妥，避免除 0，加一个 eps
    eps = 1e-6
    scale = np.linalg.norm(joints_centered.reshape(N, -1, 3), axis=-1).mean(axis=-1)  # (N,)
    scale = np.maximum(scale, eps).reshape(N, 1, 1, 1)
    joints_norm = joints_centered / scale

    # pad / 截断到 max_frames
    T_cur = joints_norm.shape[1]
    if T_cur >= max_frames:
        joints_padded = joints_norm[:, :max_frames]
    else:
        pad_len = max_frames - T_cur
        pad = np.zeros((N, pad_len, J, 3), dtype=joints_norm.dtype)
        joints_padded = np.concatenate([joints_norm, pad], axis=1)

    print("After pad:", joints_padded.shape)
    return joints_padded  # (N, max_frames, J, 3)

def make_dummy_labels(num_samples):
    """
    先给所有样本一个假标签 0，
    只是为了验证 pipeline 能跑通。
    之后你可以根据自己的动作类别替换这里。
    """
    labels = np.zeros(num_samples, dtype=int)
    return labels.tolist()

def main():
    if not os.path.exists(NPZ_PATH):
        raise FileNotFoundError(f"{NPZ_PATH} not found in current directory.")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 读取 joints 数据
    joints = load_joints_from_npz(NPZ_PATH)     # (N, T, J, C)
    N, T, J, C = joints.shape

    # 2. 归一化+对齐帧数
    joints_proc = normalize_and_pad(joints, MAX_FRAMES)  # (N, MAX_FRAMES, J, 3)

    # 3. 生成标签（先用假标签）
    labels = make_dummy_labels(N)

    # 4. 划分 train / val
    idx = np.arange(N)

    if N >= 2:
        # 正常情况：样本数 >= 2，才做 8:2 划分
        train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=True)
    else:
        # 特殊情况：只有 1 条数据，就都当成 train 和 val
        print("Warning: only 1 sample found, use the same sample for both train and val.")
        train_idx = idx
        val_idx = idx


    train_data = joints_proc[train_idx]
    val_data   = joints_proc[val_idx]

    train_labels = [labels[i] for i in train_idx]
    val_labels   = [labels[i] for i in val_idx]

    print("Train data shape:", train_data.shape)
    print("Val   data shape:", val_data.shape)
    print("Train labels:", len(train_labels))
    print("Val   labels:", len(val_labels))

    # 5. 保存为 SkateFormer 兼容格式
    np.save(os.path.join(OUT_DIR, "train_data_joint.npy"), train_data)
    np.save(os.path.join(OUT_DIR, "val_data_joint.npy"),   val_data)

    with open(os.path.join(OUT_DIR, "train_label.pkl"), "wb") as f:
        pickle.dump(train_labels, f)

    with open(os.path.join(OUT_DIR, "val_label.pkl"), "wb") as f:
        pickle.dump(val_labels, f)

    print("Saved to directory:", OUT_DIR)

if __name__ == "__main__":
    main()

