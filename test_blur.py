import cv2
from blur_detection import is_blurry

image = cv2.imread("test.jpg")

if image is None:
    print("Image not found.")
    exit()

if is_blurry(image):
    print("❌ Blurry")
else:
    print("✅ Sharp")
    