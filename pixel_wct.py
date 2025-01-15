import cv2
import numpy as np
import torch

def image_to_tensor(img):
    """Convert RGB image (H, W, 3) to tensor (1, 3, H, W)"""
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img

def tensor_to_image(tensor):
    """Convert tensor (1, 3, H, W) to RGB image (H, W, 3)"""
    tensor = tensor.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return (tensor * 255).astype(np.uint8)

def whiten_and_color(cF, sF):
    """Apply WCT transformation to RGB feature tensors"""
    cF = cF.squeeze(0)
    sF = sF.squeeze(0)
    
    # Flatten spatial dimensions
    c_mean = cF.mean(dim=(1, 2), keepdim=True)
    cF_flat = (cF - c_mean).view(cF.shape[0], -1)

    # Whitening
    c_cov = cF_flat @ cF_flat.t() / (cF_flat.shape[1] - 1)
    c_u, c_e, _ = torch.svd(c_cov)
    k_c = (c_e > 1e-5).sum()
    c_d = torch.diag(c_e[:k_c].pow(-0.5))
    whitened = c_u[:, :k_c] @ c_d @ c_u[:, :k_c].t() @ cF_flat

    # Style
    s_mean = sF.mean(dim=(1, 2), keepdim=True)
    sF_flat = (sF - s_mean).view(sF.shape[0], -1)
    s_cov = sF_flat @ sF_flat.t() / (sF_flat.shape[1] - 1)
    s_u, s_e, _ = torch.svd(s_cov)
    k_s = (s_e > 1e-5).sum()
    s_d = torch.diag(s_e[:k_s].pow(0.5))
    colored = s_u[:, :k_s] @ s_d @ s_u[:, :k_s].t() @ whitened

    # Reshape and restore mean
    out = colored.view_as(cF) + s_mean
    return out.unsqueeze(0)

def resize_style(style, content_shape):
    """Resize style image to match content dimensions"""
    return cv2.resize(style, (content_shape[1], content_shape[0]), interpolation=cv2.INTER_CUBIC)

def separate_alpha(img):
    """Separate RGB and alpha channels from image"""
    if len(img.shape) == 3 and img.shape[2] == 4:
        return img[:, :, :3], img[:, :, 3]
    else:
        return img, None

def combine_alpha(rgb_img, alpha_channel):
    """Combine RGB image with alpha channel"""
    if alpha_channel is not None:
        return np.dstack([rgb_img, alpha_channel])
    else:
        return rgb_img

def wct_color_transfer(content_img, style_img):
    """
    Apply WCT color transfer between content and style images.
    
    Args:
        content_img: numpy array of shape (H, W, C) where C is 3 (RGB) or 4 (RGBA)
        style_img: numpy array of shape (H, W, C) where C is 3 (RGB) or 4 (RGBA)
    
    Returns:
        numpy array: Transformed image with same channel count as content_img
    """
    # Separate RGB and alpha channels
    content_rgb, content_alpha = separate_alpha(content_img)
    style_rgb, style_alpha = separate_alpha(style_img)
    
    # Ensure we're working with RGB images
    if len(content_rgb.shape) != 3 or content_rgb.shape[2] != 3:
        raise ValueError(f"Content RGB should have shape (H, W, 3), got {content_rgb.shape}")
    if len(style_rgb.shape) != 3 or style_rgb.shape[2] != 3:
        raise ValueError(f"Style RGB should have shape (H, W, 3), got {style_rgb.shape}")
    
    # Resize style to match content dimensions
    style_rgb = resize_style(style_rgb, content_rgb.shape[:2])
    
    # Convert to tensors (RGB only)
    content_tensor = image_to_tensor(content_rgb)
    style_tensor = image_to_tensor(style_rgb)
    
    # Apply WCT transformation
    with torch.no_grad():
        transformed = whiten_and_color(content_tensor, style_tensor)
    
    # Convert back to image
    result_rgb = tensor_to_image(transformed)
    
    # Recombine with original alpha channel if present
    result_img = combine_alpha(result_rgb, content_alpha)
    
    return result_img

# Example usage:
"""
# For RGB images
content = cv2.imread('content.jpg')  # Shape: (H, W, 3)
style = cv2.imread('style.jpg')      # Shape: (H, W, 3)
result = wct_color_transfer(content, style)
"""
# For RGBA images
if __name__ == "__main__":
    content = cv2.imread("content/4.jpg")
    style = cv2.imread("styles/4.jpg")
    result = wct_color_transfer(content, style)
    cv2.imwrite("barewct.png", result)  # Save the result