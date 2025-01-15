import cv2
import numpy as np
from cv2.ximgproc import guidedFilter

def guided_filter_smoothing(output_img, content_img, r=50, eps=None, save_path="smoothed_output.jpg"):
    if eps is None:
        eps = (0.02 * 255) ** 2  # 26.01
    
    # Resize content image if dimensions don't match
    if output_img.shape != content_img.shape:
        content_img = cv2.resize(content_img, (output_img.shape[1], output_img.shape[0]))
    
    # Apply guided filter
    smoothed = guidedFilter(guide=content_img, src=output_img, radius=r, eps=eps)
    
    # Save result
    return smoothed

c = cv2.imread("content.jpg")
o = cv2.imread("omashit.jpg")
cv2.imwrite("wowshit.jpg", guided_filter_smoothing(o, c, r=30))
