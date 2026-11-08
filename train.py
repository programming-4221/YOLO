import random
import shutil
from pathlib import Path

# ============ 配置 ============
RAW_DIR   = Path('input')          # 标注好的图片+txt 所在目录
OUT_DIR   = Path('dataset')           # 输出目录
VAL_RATIO = 0.2                       # 验证集比例
SEED      = 42                        # 固定随机种子，保证可复现

# ============ 创建目录结构 ============
for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

# ============ 收集所有有标签的图片 ============
img_files = [p for p in RAW_DIR.iterdir()
             if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]

# 只保留有对应 .txt 的图片，避免漏标
pairs = []
for img in img_files:
    lbl = img.with_suffix('.txt')
    if lbl.exists():
        pairs.append((img, lbl))
    else:
        print(f"⚠️ 跳过无标签图片: {img.name}")

print(f"共找到 {len(pairs)} 对有效 图片-标签")

# ============ 随机打乱并划分 ============
random.seed(SEED)
random.shuffle(pairs)

n_val  = int(len(pairs) * VAL_RATIO)
val_pairs   = pairs[:n_val]
train_pairs = pairs[n_val:]

# ============ 复制文件 ============
def copy_pairs(pairs_list, split):
    for img, lbl in pairs_list:
        shutil.copy2(img, OUT_DIR / 'images' / split / img.name)
        shutil.copy2(lbl, OUT_DIR / 'labels' / split / lbl.name)

copy_pairs(train_pairs, 'train')
copy_pairs(val_pairs,   'val')

print(f"✅ 划分完成：train={len(train_pairs)}  val={len(val_pairs)}")
print(f"   输出目录: {OUT_DIR.resolve()}")