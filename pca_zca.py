import cv2
import numpy as np

def image_to_array(img):
    return img.astype(np.float32) / 255.0

def array_to_image(arr):
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)

def pca_color_transfer(content_img, style_img, eps=1e-3):
    # Flatten to [3, H*W]
    c_flat = content_img.reshape(-1, 3).T
    s_flat = style_img.reshape(-1, 3).T

    # Means
    c_mean = np.mean(c_flat, axis=1, keepdims=True)
    s_mean = np.mean(s_flat, axis=1, keepdims=True)

    # Zero-centered
    c_centered = c_flat - c_mean
    s_centered = s_flat - s_mean

    # Covariance + eps regularization
    c_cov = np.cov(c_centered) + np.eye(3) * eps
    s_cov = np.cov(s_centered) + np.eye(3) * eps

    # Eigen decomposition with clipping
    c_eigvals, c_eigvecs = np.linalg.eigh(c_cov)
    c_eigvals = np.clip(c_eigvals, 1e-5, None)

    s_eigvals, s_eigvecs = np.linalg.eigh(s_cov)
    s_eigvals = np.clip(s_eigvals, 1e-5, None)

    # Whitening and coloring transforms
    c_d_inv_sqrt = np.diag(1.0 / np.sqrt(c_eigvals))
    s_d_sqrt = np.diag(np.sqrt(s_eigvals))

    whiten = c_eigvecs @ c_d_inv_sqrt @ c_eigvecs.T @ c_centered
    colored = s_eigvecs @ s_d_sqrt @ s_eigvecs.T @ whiten

    # Add style mean
    result = colored + s_mean

    # Reshape and clip
    result_img = result.T.reshape(content_img.shape)
    return np.clip(result_img, 0, 1)

def enhance_contrast(img, factor=1.2):
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    enhanced = (img - mean) * factor + mean
    return np.clip(enhanced, 0, 1)

def resize_style(style, content_shape):
    return cv2.resize(style, (content_shape[1], content_shape[0]), interpolation=cv2.INTER_CUBIC)

def pca_color_transfer_pipeline(content_img, style_img, enhance_style=True):
    style_img = resize_style(style_img, content_img.shape[:2])

    if enhance_style:
        style_img = enhance_contrast(style_img)

    content_arr = image_to_array(content_img)
    style_arr = image_to_array(style_img)

    transferred = pca_color_transfer(content_arr, style_arr)

    return array_to_image(transferred)

if __name__ == "__main__":
    content = cv2.imread("content/4.jpg")
    style = cv2.imread("styles/4.jpg")

    content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)

    result = pca_color_transfer_pipeline(content, style)

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("pca_color_transfer_fixed.png", result_bgr)

    print("PCA color transfer completed! Output saved as 'pca_color_transfer_fixed.png'")
