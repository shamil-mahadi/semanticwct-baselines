import cv2
import numpy as np
import argparse
import os

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img

def save_image(path, img):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def compute_luminance(img):
    # img: [H, W, 3] RGB float32 [0,1]
    return np.clip(0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2], 0, 1)

def luminance_aware_blend(content_img, stylized_img, gamma=0.7, strength=1.0):
    """
    strength: float between 0 (no stylization) and 1 (full stylization in sky areas)
    """
    luminance = compute_luminance(content_img)
    alpha = (luminance ** gamma) * strength
    alpha = np.expand_dims(np.clip(alpha, 0, 1), axis=2)  # shape [H, W, 1]

    return (1 - alpha) * content_img + alpha * stylized_img

def restore_luminance(content_img, stylized_img):
    # Convert to YUV to replace Y channel
    content_yuv = cv2.cvtColor((content_img * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    stylized_yuv = cv2.cvtColor((stylized_img * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    stylized_yuv[..., 0] = content_yuv[..., 0]
    restored = cv2.cvtColor(stylized_yuv, cv2.COLOR_YUV2RGB).astype(np.float32) / 255.0
    return restored

def main(content_path, stylized_path, output_path):
    content = load_image(content_path)
    stylized = load_image(stylized_path)

    # Resize stylized image to match content image resolution
    stylized = cv2.resize(stylized, (content.shape[1], content.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Step 1: Blend based on luminance
    blended = luminance_aware_blend(content, stylized, gamma=2, strength=5)

    # Step 2 (Optional): Restore contrast
    final = restore_luminance(content, blended)

    # Save output
    save_image(output_path, final)
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--stylized", required=True, help="Path to stylized image")
    parser.add_argument("--output", default="result.jpg", help="Output image path")
    args = parser.parse_args()

    main(args.content, args.stylized, args.output)
