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

def adain(content_features, style_features, alpha=1.0):
    """
    Apply Adaptive Instance Normalization (AdaIN)
    
    Args:
        content_features: tensor of shape (1, C, H, W)
        style_features: tensor of shape (1, C, H, W)
        alpha: stylization strength (0.0 = no style, 1.0 = full style)
    
    Returns:
        tensor: AdaIN transformed features
    """
    # Calculate statistics
    content_mean = content_features.mean(dim=(2, 3), keepdim=True)
    content_std = content_features.std(dim=(2, 3), keepdim=True)
    
    style_mean = style_features.mean(dim=(2, 3), keepdim=True)
    style_std = style_features.std(dim=(2, 3), keepdim=True)
    
    # Normalize content features
    normalized = (content_features - content_mean) / (content_std + 1e-8)
    
    # Apply style statistics with controllable strength
    stylized = normalized * (alpha * style_std + (1 - alpha) * content_std) + (alpha * style_mean + (1 - alpha) * content_mean)
    
    return stylized

def enhance_contrast(img, factor=1.2):
    """Enhance image contrast"""
    img_float = img.astype(np.float32) / 255.0
    mean = np.mean(img_float)
    enhanced = (img_float - mean) * factor + mean
    enhanced = np.clip(enhanced, 0, 1)
    return (enhanced * 255).astype(np.uint8)

def resize_style(style, content_shape):
    """Resize style image to match content dimensions"""
    return cv2.resize(style, (content_shape[1], content_shape[0]), interpolation=cv2.INTER_CUBIC)

def adain_color_transfer(content_img, style_img, alpha=1.5, enhance_style=True):
    """
    Apply AdaIN color transfer between content and style images.
    
    Args:
        content_img: numpy array of shape (H, W, 3) RGB
        style_img: numpy array of shape (H, W, 3) RGB
        alpha: stylization strength (>1.0 for stronger style)
        enhance_style: whether to enhance style contrast before transfer
    
    Returns:
        numpy array: Transformed RGB image
    """
    # Ensure we're working with RGB images
    if len(content_img.shape) != 3 or content_img.shape[2] != 3:
        raise ValueError(f"Content should have shape (H, W, 3), got {content_img.shape}")
    if len(style_img.shape) != 3 or style_img.shape[2] != 3:
        raise ValueError(f"Style should have shape (H, W, 3), got {style_img.shape}")
    
    # Resize style to match content dimensions
    style_img = resize_style(style_img, content_img.shape[:2])
    
    # Optional: Enhance style contrast for stronger transfer
    if enhance_style:
        style_img = enhance_contrast(style_img, factor=1.3)
    
    # Convert to tensors
    content_tensor = image_to_tensor(content_img)
    style_tensor = image_to_tensor(style_img)
    
    # Apply AdaIN transformation with controllable strength
    with torch.no_grad():
        transformed = adain(content_tensor, style_tensor, alpha=alpha)
    
    # Convert back to image
    result_img = tensor_to_image(transformed)
    
    return result_img

if __name__ == "__main__":
    # Load images
    content = cv2.imread("content/4.jpg")
    style = cv2.imread("styles/4.jpg")
    
    # Convert from BGR to RGB (OpenCV loads as BGR)
    content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
    style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
    
    # Apply AdaIN color transfer with different strengths
    result_weak = adain_color_transfer(content, style, alpha=0.8)
    result_normal = adain_color_transfer(content, style, alpha=1.5)
    result_strong = adain_color_transfer(content, style, alpha=2.0)
    
    # Convert back to BGR for saving with OpenCV
    result_weak = cv2.cvtColor(result_weak, cv2.COLOR_RGB2BGR)
    result_normal = cv2.cvtColor(result_normal, cv2.COLOR_RGB2BGR)
    result_strong = cv2.cvtColor(result_strong, cv2.COLOR_RGB2BGR)
    
    # Save the results
    cv2.imwrite("adain_weak.png", result_weak)
    cv2.imwrite("adain_normal.png", result_normal)
    cv2.imwrite("adain_strong.png", result_strong)
    print("AdaIN color transfer completed! Results saved as 'adain_weak.png', 'adain_normal.png', and 'adain_strong.png'")
    
    # Optional: Display results (uncomment if you want to see them)
    # cv2.imshow('Content', cv2.cvtColor(content, cv2.COLOR_RGB2BGR))
    # cv2.imshow('Style', cv2.cvtColor(style, cv2.COLOR_RGB2BGR))
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()