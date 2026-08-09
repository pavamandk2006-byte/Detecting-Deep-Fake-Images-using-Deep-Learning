import os
import cv2
import logging
from tqdm import tqdm

from config import REAL_VIDEO_DIR, FAKE_VIDEO_DIR, FRAME_SKIP
from blur_detection import is_blurry
from face_detector import detect_face
from image_utils import save_face


# ==========================================================
# LOGGING
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


# ==========================================================
# PROCESS ONE VIDEO
# ==========================================================

def process_video(video_path, label):
    """
    Extract every 15th frame from a video,
    detect the face, remove blurry faces,
    and save valid face images.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.warning(f"Could not open video: {video_path}")
        return 0

    frame_number = 0
    saved_faces = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Extract every 15th frame
        if frame_number % FRAME_SKIP != 0:
            frame_number += 1
            continue

        # Detect largest face
        face = detect_face(frame)

        if face is None or face.size == 0:
            frame_number += 1
            continue

        # Reject blurry faces
        if is_blurry(face):
            frame_number += 1
            continue

        # Save face
        save_face(
            face=face,
            label=label
        )

        saved_faces += 1
        frame_number += 1

    cap.release()

    return saved_faces


# ==========================================================
# PROCESS VIDEO DIRECTORY
# ==========================================================

def process_directory(video_directory, label):

    if not os.path.isdir(video_directory):
        raise FileNotFoundError(
            f"Video directory does not exist: {video_directory}"
        )

    video_files = [
        file
        for file in os.listdir(video_directory)
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    video_files.sort()

    logging.info(
        f"Found {len(video_files)} {label} videos."
    )

    total_faces = 0

    for filename in tqdm(
        video_files,
        desc=f"Processing {label}"
    ):

        video_path = os.path.join(
            video_directory,
            filename
        )

        try:

            saved = process_video(
                video_path,
                label
            )

            total_faces += saved

        except Exception as error:

            logging.exception(
                f"Error processing {filename}: {error}"
            )

    logging.info(
        f"{label}: saved {total_faces} face images."
    )

    return total_faces


# ==========================================================
# MAIN
# ==========================================================

def main():

    logging.info("=" * 60)
    logging.info("FaceForensics++ Face Extraction")
    logging.info("=" * 60)

    logging.info(f"Frame skip: {FRAME_SKIP}")

    # ------------------------------------------
    # REAL
    # ------------------------------------------

    real_faces = process_directory(
        REAL_VIDEO_DIR,
        "real"
    )

    # ------------------------------------------
    # FAKE
    # ------------------------------------------

    fake_faces = process_directory(
        FAKE_VIDEO_DIR,
        "fake"
    )

    # ------------------------------------------
    # RESULTS
    # ------------------------------------------

    logging.info("=" * 60)

    logging.info(
        f"Real faces extracted: {real_faces}"
    )

    logging.info(
        f"Fake faces extracted: {fake_faces}"
    )

    logging.info(
        f"Total faces extracted: "
        f"{real_faces + fake_faces}"
    )

    logging.info("=" * 60)
    logging.info("Face extraction completed.")


if __name__ == "__main__":
    main()