import cv2
from config import BLUR_THRESHOLD


def is_blurry(image):
    """
    Check whether an image is blurry using the
    variance of the Laplacian.

    Returns:
        True  -> Blurry image
        False -> Sharp image
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return score < BLUR_THRESHOLD