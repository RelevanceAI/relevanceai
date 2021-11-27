from PIL import Image
import numpy as np
import base64
from io import BytesIO
import cv2
import numpy as np

def resize(
    img,
    width = None,
    height = None):
    (h, w) = img.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    return resized


def numpy_to_b64(array: np.ndarray, scalar: bool = True) -> str:
    '''
    Convert from 0-1 to 0-255
    '''
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    return im_b64
