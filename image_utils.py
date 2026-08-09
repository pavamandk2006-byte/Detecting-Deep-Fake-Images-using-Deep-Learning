import os
import uuid
import random
import cv2

from config import *

def save_face(face, label):
    """
    Save detected face to train/ or val/.
    """

    if label == "real":

        if random.random() < TRAIN_SPLIT:
            save_dir = TRAIN_REAL_DIR
        else:
            save_dir = VAL_REAL_DIR

    else:

        if random.random() < TRAIN_SPLIT:
            save_dir = TRAIN_FAKE_DIR
        else:
            save_dir = VAL_FAKE_DIR

    filename = f"{uuid.uuid4().hex}.jpg"

    save_path = os.path.join(
        save_dir,
        filename
    )

    face = cv2.resize(
        face,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    cv2.imwrite(
        save_path,
        face
    )