import cv2
from face_detector import detect_face

image = cv2.imread("test.jpg")

if image is None:
    print("❌ test.jpg not found.")
    exit()

face = detect_face(image)

if face is None:
    print("❌ No face detected.")
else:
    print("✅ Face detected!")

    cv2.imshow("Detected Face", face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()