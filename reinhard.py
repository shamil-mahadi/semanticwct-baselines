import cv2
import numpy as np
import os

def lab_color_transfer(content, style, alpha=1.5):
    content_lab = cv2.cvtColor(content, cv2.COLOR_RGB2LAB).astype(np.float32)
    style_lab = cv2.cvtColor(style, cv2.COLOR_RGB2LAB).astype(np.float32)

    c_l, c_a, c_b = cv2.split(content_lab)
    s_l, s_a, s_b = cv2.split(style_lab)

    def match(c_channel, s_channel):
        c_mean, c_std = c_channel.mean(), c_channel.std()
        s_mean, s_std = s_channel.mean(), s_channel.std()
        return ((c_channel - c_mean) * (s_std / (c_std + 1e-5)) + s_mean)

    matched_l = match(c_l, s_l)
    matched_a = match(c_a, s_a)
    matched_b = match(c_b, s_b)

    # Chroma scaling for stylization strength
    mean_a = matched_a.mean()
    mean_b = matched_b.mean()
    matched_a = (matched_a - mean_a) * alpha + mean_a
    matched_b = (matched_b - mean_b) * alpha + mean_b

    transferred = cv2.merge([matched_l, matched_a, matched_b])
    transferred = np.clip(transferred, 0, 255).astype(np.uint8)
    return cv2.cvtColor(transferred, cv2.COLOR_LAB2RGB)

def resize_style(style_img, target_shape):
    return cv2.resize(style_img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)

def make_output_dirs():
    os.makedirs('reinhardoutput/full', exist_ok=True)
    os.makedirs('reinhardoutput/compare', exist_ok=True)

def create_comparison_image(content, style, output):
    h = max(content.shape[0], style.shape[0], output.shape[0])
    w_c = content.shape[1]
    w_s = style.shape[1]
    w_o = output.shape[1]
    canvas = np.ones((h, w_c + w_s + w_o + 40, 3), dtype=np.uint8) * 255

    canvas[:content.shape[0], :w_c] = content
    canvas[:style.shape[0], w_c + 20:w_c + 20 + w_s] = style
    canvas[:output.shape[0], w_c + w_s + 40:] = output

    return canvas

def process_all_images(alpha=1.5):
    make_output_dirs()

    for i in range(1, 11):
        content_path = f"content/{i}.jpg"
        style_path = f"styles/{i}.jpg"

        if not (os.path.exists(content_path) and os.path.exists(style_path)):
            print(f"Skipping {i}.jpg — missing content or style.")
            continue

        # Load images and convert to RGB
        content = cv2.cvtColor(cv2.imread(content_path), cv2.COLOR_BGR2RGB)
        style = cv2.cvtColor(cv2.imread(style_path), cv2.COLOR_BGR2RGB)

        # Resize style for preview/comparison
        style_resized = resize_style(style, content.shape[:2])

        # Apply color transfer
        output = lab_color_transfer(content, style_resized, alpha=alpha)

        # Save full-res output
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"reinhardoutput/full/{i}.jpg", output_bgr)

        # Save side-by-side comparison (content | style | output)
        comparison = create_comparison_image(
            cv2.cvtColor(content, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(style_resized, cv2.COLOR_RGB2BGR),
            output_bgr
        )
        cv2.imwrite(f"reinhardoutput/compare/{i}_compare.jpg", comparison)

        print(f"Processed and saved image {i}.jpg")

if __name__ == "__main__":
    process_all_images(alpha=1.0)
    print("Batch Reinhard color transfer completed!")
