import os
import shutil
import random

# ✅ Set correct source (just the parent 'test' folder)
source_dir = r"C:\Users\bharg\Downloads\Cleantech-Transforming-Waste-management-with-Transfer-learning-main\Cleantech-Transforming-Waste-management-with-Transfer-learning-main\Waste_Classification_Dataset\test"

# ✅ Set destination somewhere separate (e.g. 'split_dataset')
dest_dir = r"C:\Users\bharg\Downloads\Cleantech-Transforming-Waste-management-with-Transfer-learning-main\split_dataset"

split_ratio = 0.8  # 80% for training

classes = ['biodegradable', 'recyclable', 'trash']

# Create destination folders
for split in ['train', 'test']:
    for cls in classes:
        os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

# Split and copy files
for cls in classes:
    cls_dir = os.path.join(source_dir, cls)

    if not os.path.exists(cls_dir):
        print(f"⚠️ Skipping missing class folder: {cls_dir}")
        continue

    images = os.listdir(cls_dir)
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_imgs = images[:split_point]
    test_imgs = images[split_point:]

    for img in train_imgs:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(dest_dir, 'train', cls, img)
        shutil.copyfile(src, dst)

    for img in test_imgs:
        src = os.path.join(cls_dir, img)
        dst = os.path.join(dest_dir, 'test', cls, img)
        shutil.copyfile(src, dst)

print("✅ Dataset split into train and test folders successfully.")
