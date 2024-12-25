import cv2
import numpy as np

def preprocess_image(image_path, resize_dim=(200, 50), debug=False):
    """
    Preprocesses an image for OCR or analysis.

    :param image_path: Path to the input image.
    :param resize_dim: Tuple for resizing the image (width, height).
    :param debug: Boolean to save debug images.
    :return: Preprocessed image as a numpy array.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Resize the image
    resized = cv2.resize(binary, resize_dim, interpolation=cv2.INTER_AREA)

    # Save debug image if requested
    if debug:
        debug_path = "debug_preprocessed.png"
        cv2.imwrite(debug_path, resized)

    return resized
