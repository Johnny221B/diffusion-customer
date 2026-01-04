from PIL import Image
import numpy as np

def make_canny_edge(ref_img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    import cv2  # 如果报错：pip install opencv-python

    img = np.array(ref_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=low, threshold2=high)
    edges_3ch = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_3ch)
