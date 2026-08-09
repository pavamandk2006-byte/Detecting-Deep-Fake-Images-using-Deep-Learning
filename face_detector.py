import os
import cv2

from config import (
    PROTOTXT,
    CAFFE_MODEL,
    CONFIDENCE_THRESHOLD,
    MIN_FACE_SIZE,
)


# ==========================================================
# LOAD OPENCV DNN FACE DETECTOR
# ==========================================================

if not os.path.exists(PROTOTXT):
    raise FileNotFoundError(
        f"Missing OpenCV prototxt file:\n{PROTOTXT}"
    )


if not os.path.exists(CAFFE_MODEL):
    raise FileNotFoundError(
        f"Missing OpenCV Caffe model:\n{CAFFE_MODEL}"
    )


face_net = cv2.dnn.readNetFromCaffe(
    PROTOTXT,
    CAFFE_MODEL,
)


# ==========================================================
# DETECT ALL FACES
# ==========================================================

def detect_faces(frame):
    """
    Detect ALL faces in an image.

    Parameters
    ----------
    frame : numpy.ndarray
        Original OpenCV BGR image.

    Returns
    -------
    list
        A list containing every detected face.

    Example:

    [
        {
            "x1": 100,
            "y1": 120,
            "x2": 350,
            "y2": 450,
            "confidence": 0.98
        }
    ]

    The original image is NOT resized.
    Only the internal detector input is resized
    to 300x300.
    """

    if frame is None:
        return []


    if frame.size == 0:
        return []


    # ======================================================
    # ORIGINAL IMAGE SIZE
    # ======================================================

    height, width = frame.shape[:2]


    # ======================================================
    # CREATE DNN INPUT
    # ======================================================

    blob = cv2.dnn.blobFromImage(
        cv2.resize(
            frame,
            (300, 300),
        ),

        1.0,

        (300, 300),

        (104, 177, 123),
    )


    # ======================================================
    # RUN FACE DETECTOR
    # ======================================================

    face_net.setInput(blob)

    detections = face_net.forward()


    faces = []


    # ======================================================
    # READ EVERY DETECTION
    # ======================================================

    for i in range(
        detections.shape[2]
    ):

        confidence = float(
            detections[
                0,
                0,
                i,
                2
            ]
        )


        # --------------------------------------------------
        # CONFIDENCE FILTER
        # --------------------------------------------------

        if (
            confidence
            < CONFIDENCE_THRESHOLD
        ):
            continue


        # --------------------------------------------------
        # GET BOUNDING BOX
        # --------------------------------------------------

        box = (
            detections[
                0,
                0,
                i,
                3:7
            ]
            * [width, height, width, height]
        )


        x1, y1, x2, y2 = (
            box.astype(int)
        )


        # --------------------------------------------------
        # KEEP BOX INSIDE IMAGE
        # --------------------------------------------------

        x1 = max(
            0,
            x1,
        )

        y1 = max(
            0,
            y1,
        )

        x2 = min(
            width,
            x2,
        )

        y2 = min(
            height,
            y2,
        )


        # --------------------------------------------------
        # FACE SIZE
        # --------------------------------------------------

        face_width = (
            x2 - x1
        )

        face_height = (
            y2 - y1
        )


        if (
            face_width < MIN_FACE_SIZE
            or
            face_height < MIN_FACE_SIZE
        ):
            continue


        # --------------------------------------------------
        # SAVE DETECTION
        # --------------------------------------------------

        faces.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": confidence,
            }
        )


    # ======================================================
    # REMOVE DUPLICATE / OVERLAPPING FACES
    # ======================================================

    faces = remove_duplicate_faces(
        faces
    )


    # ======================================================
    # SORT FACES
    # ======================================================

    # Sort from top-left to bottom-right.

    faces.sort(
        key=lambda face: (
            face["y1"],
            face["x1"],
        )
    )


    return faces


# ==========================================================
# REMOVE DUPLICATE DETECTIONS
# ==========================================================

def remove_duplicate_faces(
    faces,
    overlap_threshold=0.50,
):
    """
    Removes heavily overlapping detections.

    This prevents the same face from being
    processed multiple times.
    """

    if len(faces) <= 1:
        return faces


    # Sort by confidence.

    faces = sorted(
        faces,
        key=lambda face:
            face["confidence"],
        reverse=True,
    )


    selected = []


    for current in faces:

        duplicate = False


        for existing in selected:

            overlap = calculate_iou(
                current,
                existing,
            )


            if (
                overlap
                >= overlap_threshold
            ):

                duplicate = True

                break


        if not duplicate:

            selected.append(
                current
            )


    return selected


# ==========================================================
# INTERSECTION OVER UNION
# ==========================================================

def calculate_iou(
    box_a,
    box_b,
):
    """
    Calculate Intersection over Union
    between two bounding boxes.
    """

    ax1 = box_a["x1"]
    ay1 = box_a["y1"]
    ax2 = box_a["x2"]
    ay2 = box_a["y2"]


    bx1 = box_b["x1"]
    by1 = box_b["y1"]
    bx2 = box_b["x2"]
    by2 = box_b["y2"]


    # Intersection coordinates

    ix1 = max(
        ax1,
        bx1,
    )

    iy1 = max(
        ay1,
        by1,
    )

    ix2 = min(
        ax2,
        bx2,
    )

    iy2 = min(
        ay2,
        by2,
    )


    intersection_width = max(
        0,
        ix2 - ix1,
    )


    intersection_height = max(
        0,
        iy2 - iy1,
    )


    intersection_area = (
        intersection_width
        * intersection_height
    )


    # Areas

    area_a = (
        max(0, ax2 - ax1)
        *
        max(0, ay2 - ay1)
    )


    area_b = (
        max(0, bx2 - bx1)
        *
        max(0, by2 - by1)
    )


    union_area = (
        area_a
        + area_b
        - intersection_area
    )


    if union_area <= 0:
        return 0.0


    return (
        intersection_area
        / union_area
    )


# ==========================================================
# BACKWARD COMPATIBILITY
# ==========================================================

def detect_face(frame):
    """
    Compatibility function for older code.

    IMPORTANT:
    New application code should use detect_faces()
    because it needs to process multiple faces.

    This function returns the largest detected face.
    """

    faces = detect_faces(
        frame
    )


    if not faces:
        return None


    largest = max(
        faces,
        key=lambda face:
            (
                face["x2"] - face["x1"]
            )
            *
            (
                face["y2"] - face["y1"]
            ),
    )


    return frame[
        largest["y1"]:largest["y2"],
        largest["x1"]:largest["x2"],
    ]


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DeepGuard AI - Face Detector Test")
    print("=" * 60)

    print(
        "Detector loaded successfully."
    )

    print(
        f"Confidence threshold : "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"Minimum face size    : "
        f"{MIN_FACE_SIZE}px"
    )

    print("=" * 60)