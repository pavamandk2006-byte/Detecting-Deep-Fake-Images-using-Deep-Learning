from PIL import Image
import os

folders = [
    "dataset/train/real",
    "dataset/train/fake",
    "dataset/val/real",
    "dataset/val/fake"
]

deleted = 0

for folder in folders:

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:
            with Image.open(path) as img:
                img.verify()

        except Exception:
            print("Deleting:", path)
            os.remove(path)
            deleted += 1

print()
print("Deleted", deleted, "corrupted images.")