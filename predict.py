import sys
import os
import cv2
import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.models import efficientnet_b0

from PIL import Image

from face_detector import detect_faces


# ==========================================================
# CONFIGURATION
# ==========================================================

MODEL_PATH = "checkpoints/best_model.pth"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CLASS_NAMES = {
    0: "REAL",
    1: "FAKE"
}

RESULT_FOLDER = "prediction_results"


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    model = efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

])


# ==========================================================
# PREDICT ONE FACE
# ==========================================================

def predict_face(model, face):

    # Convert OpenCV BGR → RGB
    face_rgb = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL
    image = Image.fromarray(
        face_rgb
    )

    # Resize to 224 × 224
    image = transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    # Move to CPU/GPU
    image = image.to(DEVICE)

    # ------------------------------------------------------
    # Prediction
    # ------------------------------------------------------

    with torch.no_grad():

        output = model(image)

        probabilities = torch.softmax(
            output,
            dim=1
        )

    # Scores are used internally only
    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    label = CLASS_NAMES[
        predicted_class
    ]

    real_probability = (
        probabilities[0, 0].item()
    )

    fake_probability = (
        probabilities[0, 1].item()
    )

    return (
        label,
        real_probability,
        fake_probability
    )


# ==========================================================
# DRAW RESULT
# ==========================================================

def draw_result(
    image,
    result,
    face_number
):

    x1, y1, x2, y2 = result["box"]

    label = result["label"]

    # ------------------------------------------------------
    # Only show REAL / FAKE
    # ------------------------------------------------------

    text = (
        f"Face {face_number}: {label}"
    )

    # ------------------------------------------------------
    # Box color
    # ------------------------------------------------------

    if label == "REAL":

        box_color = (
            0,
            255,
            0
        )

    else:

        box_color = (
            0,
            0,
            255
        )

    # ------------------------------------------------------
    # Thickness based on image size
    # ------------------------------------------------------

    thickness = max(
        2,
        int(
            min(
                image.shape[0],
                image.shape[1]
            ) / 500
        )
    )

    # ------------------------------------------------------
    # Draw face box
    # ------------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        box_color,
        thickness
    )

    # ------------------------------------------------------
    # Text settings
    # ------------------------------------------------------

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = max(
        0.6,
        min(
            image.shape[0],
            image.shape[1]
        ) / 1800
    )

    text_thickness = max(
        1,
        thickness
    )

    text_size = cv2.getTextSize(
        text,
        font,
        font_scale,
        text_thickness
    )[0]

    text_width = text_size[0]

    text_height = text_size[1]

    # ------------------------------------------------------
    # Text position
    # ------------------------------------------------------

    text_y = max(
        y1,
        text_height + 10
    )

    # ------------------------------------------------------
    # Text background
    # ------------------------------------------------------

    cv2.rectangle(
        image,

        (
            x1,
            text_y - text_height - 10
        ),

        (
            x1 + text_width + 10,
            text_y + 5
        ),

        box_color,

        -1
    )

    # ------------------------------------------------------
    # Draw text
    # ------------------------------------------------------

    cv2.putText(
        image,

        text,

        (
            x1 + 5,
            text_y - 3
        ),

        font,

        font_scale,

        (
            255,
            255,
            255
        ),

        text_thickness,

        cv2.LINE_AA
    )


# ==========================================================
# PREDICT IMAGE
# ==========================================================

def predict(image_path):

    print()
    print("=" * 55)
    print("             DEEPFAKE DETECTION")
    print("=" * 55)

    print()
    print(
        f"Image : {image_path}"
    )

    # ------------------------------------------------------
    # Read original image
    # ------------------------------------------------------

    frame = cv2.imread(
        image_path
    )

    if frame is None:

        print()
        print(
            "ERROR: Could not open image."
        )

        return

    # ------------------------------------------------------
    # Original resolution
    # ------------------------------------------------------

    height, width = frame.shape[:2]

    print()
    print(
        f"Resolution : {width} x {height}"
    )

    # ------------------------------------------------------
    # Detect ALL faces
    # ------------------------------------------------------

    print()
    print(
        "Detecting faces..."
    )

    faces = detect_faces(
        frame
    )

    if len(faces) == 0:

        print()
        print(
            "No faces detected."
        )

        print("=" * 55)

        return

    print()
    print(
        f"Faces detected : {len(faces)}"
    )

    # ------------------------------------------------------
    # Load model
    # ------------------------------------------------------

    print(
        "Analyzing faces..."
    )

    model = load_model()

    # ------------------------------------------------------
    # Store results
    # ------------------------------------------------------

    results = []

    # ------------------------------------------------------
    # Predict every face
    # ------------------------------------------------------

    for index, face_data in enumerate(
        faces,
        start=1
    ):

        (
            face,
            x1,
            y1,
            x2,
            y2,
            detector_confidence
        ) = face_data

        (
            label,
            real_probability,
            fake_probability
        ) = predict_face(
            model,
            face
        )

        result = {

            "face": index,

            "label": label,

            # Scores remain internal.
            "real": real_probability,

            "fake": fake_probability,

            "box": (
                x1,
                y1,
                x2,
                y2
            )
        }

        results.append(
            result
        )

    # ======================================================
    # DISPLAY SIMPLE RESULTS
    # ======================================================

    print()
    print("=" * 55)
    print("                 RESULTS")
    print("=" * 55)

    for result in results:

        print(
            f"Face {result['face']} : "
            f"{result['label']}"
        )

    # ======================================================
    # OVERALL RESULT
    # ======================================================

    # Count faces classified as FAKE
    fake_faces = sum(
        1
        for result in results
        if result["label"] == "FAKE"
    )

    real_faces = sum(
        1
        for result in results
        if result["label"] == "REAL"
    )

    total_faces = len(results)

    # ------------------------------------------------------
    # Overall decision
    #
    # If at least one face is FAKE,
    # mark the image as suspicious.
    # ------------------------------------------------------

    if fake_faces > 0:

        overall_label = "FAKE"

    else:

        overall_label = "REAL"

    # ======================================================
    # DRAW RESULTS ON ORIGINAL IMAGE
    # ======================================================

    output_image = frame.copy()

    for result in results:

        draw_result(
            output_image,
            result,
            result["face"]
        )

    # ------------------------------------------------------
    # Overall label on image
    # ------------------------------------------------------

    overall_text = (
        f"OVERALL: {overall_label}"
    )

    if overall_label == "REAL":

        overall_color = (
            0,
            255,
            0
        )

    else:

        overall_color = (
            0,
            0,
            255
        )

    cv2.putText(
        output_image,

        overall_text,

        (
            30,
            60
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.5,

        overall_color,

        4,

        cv2.LINE_AA
    )

    # ======================================================
    # SAVE RESULT
    # ======================================================

    os.makedirs(
        RESULT_FOLDER,
        exist_ok=True
    )

    filename = os.path.basename(
        image_path
    )

    output_path = os.path.join(
        RESULT_FOLDER,
        "result_" + filename
    )

    cv2.imwrite(
        output_path,
        output_image
    )

    # ======================================================
    # FINAL DISPLAY
    # ======================================================

    print()
    print("=" * 55)

    print(
        f"Total faces : {total_faces}"
    )

    print(
        f"REAL faces  : {real_faces}"
    )

    print(
        f"FAKE faces  : {fake_faces}"
    )

    print()

    print(
        f"Overall Result : {overall_label}"
    )

    print()

    print(
        "Result image:"
    )

    print(
        output_path
    )

    print("=" * 55)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print(
            'python predict.py "path_to_image.jpg"'
        )

        print()

        print("Example:")
        print(
            'python predict.py '
            '"test_images\\photo.jpg"'
        )

        sys.exit(1)

    image_path = sys.argv[1]

    predict(
        image_path
    )