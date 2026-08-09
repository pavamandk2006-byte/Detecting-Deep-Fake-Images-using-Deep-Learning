import json
import os

import torch
import torch.nn as nn

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torchvision import transforms
from torchvision.models import efficientnet_b0

from config import (
    VAL_REAL_DIR,
    VAL_FAKE_DIR
)


MODEL_PATH = (
    "checkpoints/best_model.pth"
)

METRICS_PATH = "metrics.json"

IMAGE_SIZE = 224


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


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


def collect_images():

    paths = []
    labels = []

    for folder, label in [

        (
            VAL_REAL_DIR,
            0
        ),

        (
            VAL_FAKE_DIR,
            1
        )

    ]:

        if not os.path.isdir(folder):

            continue

        for filename in sorted(
            os.listdir(folder)
        ):

            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp"
                )
            ):

                paths.append(
                    os.path.join(
                        folder,
                        filename
                    )
                )

                labels.append(
                    label
                )

    return paths, labels


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

    model = model.to(
        DEVICE
    )

    model.eval()

    return model


def main():

    paths, labels = (
        collect_images()
    )

    if not paths:

        raise RuntimeError(
            "No validation images found."
        )

    print("=" * 60)

    print(
        "DEEPGUARD AI MODEL EVALUATION"
    )

    print("=" * 60)

    print(
        f"Device : {DEVICE}"
    )

    print(
        f"Validation images : {len(paths)}"
    )

    print()

    print(
        "Evaluating existing model..."
    )

    model = load_model()

    predictions = []

    with torch.no_grad():

        for index, path in enumerate(
            paths,
            start=1
        ):

            image = Image.open(
                path
            ).convert("RGB")

            tensor = transform(
                image
            )

            tensor = tensor.unsqueeze(
                0
            )

            tensor = tensor.to(
                DEVICE
            )

            output = model(
                tensor
            )

            prediction = int(
                torch.argmax(
                    output,
                    dim=1
                ).item()
            )

            predictions.append(
                prediction
            )

            if index % 500 == 0:

                print(
                    f"Processed {index}/{len(paths)}"
                )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[
            0,
            1
        ]
    )

    metrics = {

        "accuracy": float(
            accuracy
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1_score": float(
            f1
        ),

        "confusion_matrix":
            matrix.tolist(),

        "validation_images":
            len(paths),

        "real_validation_images":
            labels.count(0),

        "fake_validation_images":
            labels.count(1)

    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print()

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print()

    print(
        "Confusion Matrix:"
    )

    print(
        matrix
    )

    print()

    print(
        f"Saved: {METRICS_PATH}"
    )


if __name__ == "__main__":

    main()