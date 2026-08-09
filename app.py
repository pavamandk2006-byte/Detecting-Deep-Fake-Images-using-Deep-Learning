import os
import uuid

import cv2
import torch
import torch.nn as nn

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

from config import (
    DEVICE,
    IMAGE_SIZE,
    BEST_MODEL_PATH,
    UPLOAD_FOLDER,
    NUM_CLASSES,
)

from face_detector import detect_faces


# ==========================================================
# FLASK
# ==========================================================

app = Flask(__name__)

app.secret_key = "deepguard-project"

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_FOLDER = os.path.join(
    BASE_DIR,
    "prediction_results"
)

RESULT_IMAGE_FOLDER = os.path.join(
    RESULT_FOLDER,
    "result"
)


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RESULT_IMAGE_FOLDER,
    exist_ok=True
)


# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = {
    0: "REAL",
    1: "FAKE"
}


# ==========================================================
# IMAGE TRANSFORM
# ==========================================================

transform = transforms.Compose([
    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor()
])


# ==========================================================
# LOAD MODEL
# ==========================================================

print()
print("=" * 55)
print("              DEEPGUARD")
print("       Deepfake Image Detection")
print("=" * 55)


if not os.path.exists(BEST_MODEL_PATH):

    raise FileNotFoundError(
        "Trained model not found:\n"
        + BEST_MODEL_PATH
    )


print("Loading EfficientNet-B0...")


model = efficientnet_b0(
    weights=None
)


model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)


model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )
)


model = model.to(
    DEVICE
)

model.eval()


print("Model loaded successfully.")

print(
    "Device:",
    DEVICE
)

print("=" * 55)


# ==========================================================
# PREDICT ONE FACE
# ==========================================================

def predict_face(face):

    if face is None:
        return None

    if face.size == 0:
        return None


    # OpenCV BGR → RGB

    rgb_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )


    image = Image.fromarray(
        rgb_face
    )


    image = transform(
        image
    )


    image = image.unsqueeze(
        0
    )


    image = image.to(
        DEVICE
    )


    with torch.no_grad():

        output = model(
            image
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )


    predicted = torch.argmax(
        probabilities,
        dim=1
    ).item()


    confidence = (
        probabilities[
            0,
            predicted
        ].item()
        * 100
    )


    return {
        "label":
            CLASS_NAMES[predicted],

        "confidence":
            confidence
    }


# ==========================================================
# DRAW RESULT
# ==========================================================

def draw_result(
    image,
    x1,
    y1,
    x2,
    y2,
    prediction
):

    label = prediction["label"]


    if label == "REAL":

        color = (
            0,
            180,
            80
        )

    else:

        color = (
            0,
            0,
            220
        )


    # Face box

    cv2.rectangle(
        image,

        (
            x1,
            y1
        ),

        (
            x2,
            y2
        ),

        color,

        3
    )


    # Only show REAL / FAKE
    # to the viewer.

    text = label


    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.9

    thickness = 2


    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )


    label_y = max(
        y1,
        text_height + 15
    )


    # Background

    cv2.rectangle(
        image,

        (
            x1,
            label_y
            - text_height
            - 12
        ),

        (
            x1
            + text_width
            + 16,

            label_y
            + baseline
            - 5
        ),

        color,

        -1
    )


    # Text

    cv2.putText(
        image,

        text,

        (
            x1 + 8,
            label_y - 7
        ),

        font,

        font_scale,

        (
            255,
            255,
            255
        ),

        thickness,

        cv2.LINE_AA
    )


# ==========================================================
# SERVE RESULT IMAGE
# ==========================================================

@app.route(
    "/prediction-results/<path:filename>"
)
def result_file(filename):

    return send_from_directory(
        RESULT_IMAGE_FOLDER,
        filename
    )


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():

    return render_template(
        "home.html"
    )


# ==========================================================
# UPLOAD PAGE
# ==========================================================

@app.route("/detect")
def detect_page():

    return render_template(
        "detect.html"
    )


# ==========================================================
# ANALYZE IMAGE
# ==========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    # ------------------------------------------------------
    # CHECK FILE
    # ------------------------------------------------------

    if "image" not in request.files:

        flash(
            "Please select an image."
        )

        return redirect(
            url_for(
                "detect_page"
            )
        )


    file = request.files["image"]


    if file.filename == "":

        flash(
            "Please select an image."
        )

        return redirect(
            url_for(
                "detect_page"
            )
        )


    # ------------------------------------------------------
    # CHECK EXTENSION
    # ------------------------------------------------------

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }


    extension = os.path.splitext(
        file.filename
    )[1].lower()


    if extension not in allowed_extensions:

        flash(
            "Please upload a JPG, JPEG, PNG or WEBP image."
        )

        return redirect(
            url_for(
                "detect_page"
            )
        )


    # ------------------------------------------------------
    # SAVE UPLOAD
    # ------------------------------------------------------

    unique_name = (
        uuid.uuid4().hex
        + extension
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        unique_name
    )


    file.save(
        upload_path
    )


    # ------------------------------------------------------
    # READ IMAGE
    # ------------------------------------------------------

    image = cv2.imread(
        upload_path
    )


    if image is None:

        flash(
            "Unable to read the uploaded image."
        )

        return redirect(
            url_for(
                "detect_page"
            )
        )


    height, width = (
        image.shape[:2]
    )


    print()
    print("=" * 55)

    print(
        "Image:",
        file.filename
    )

    print(
        "Resolution:",
        width,
        "x",
        height
    )

    print(
        "Detecting faces..."
    )


    # ------------------------------------------------------
    # DETECT ALL FACES
    # ------------------------------------------------------

    faces = detect_faces(
        image
    )


    print(
        "Faces detected:",
        len(faces)
    )


    # ------------------------------------------------------
    # IF NO FACE
    # ------------------------------------------------------

    if len(faces) == 0:

        overall_result = (
            "NO FACE DETECTED"
        )

        output_image = image.copy()


    else:

        output_image = image.copy()

        predictions = []


        # --------------------------------------------------
        # PREDICT EVERY FACE
        # --------------------------------------------------

        for face in faces:

            x1 = int(
                face["x1"]
            )

            y1 = int(
                face["y1"]
            )

            x2 = int(
                face["x2"]
            )

            y2 = int(
                face["y2"]
            )


            cropped_face = image[
                y1:y2,
                x1:x2
            ]


            prediction = predict_face(
                cropped_face
            )


            if prediction is None:
                continue


            predictions.append(
                prediction
            )


            draw_result(
                output_image,

                x1,
                y1,

                x2,
                y2,

                prediction
            )


        # --------------------------------------------------
        # OVERALL RESULT
        # --------------------------------------------------

        if len(predictions) == 0:

            overall_result = (
                "NO FACE DETECTED"
            )


        else:

            real_count = sum(
                1
                for prediction
                in predictions
                if prediction["label"]
                == "REAL"
            )


            fake_count = sum(
                1
                for prediction
                in predictions
                if prediction["label"]
                == "FAKE"
            )


            # Majority decision

            if fake_count > real_count:

                overall_result = "FAKE"

            else:

                overall_result = "REAL"


    # ------------------------------------------------------
    # SAVE RESULT IMAGE
    # ------------------------------------------------------

    result_filename = (
        "result_"
        + uuid.uuid4().hex
        + ".jpg"
    )


    result_path = os.path.join(
        RESULT_IMAGE_FOLDER,
        result_filename
    )


    cv2.imwrite(
        result_path,
        output_image
    )


    print(
        "Prediction:",
        overall_result
    )

    print(
        "Result image:",
        result_path
    )

    print("=" * 55)


    # ------------------------------------------------------
    # RESULT PAGE
    # ------------------------------------------------------

    return render_template(
        "result.html",

        result={
            "overall_result":
                overall_result,

            "result_image":
                result_filename
        }
    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    print()
    print(
        "Open your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )