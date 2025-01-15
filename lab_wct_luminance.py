import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import json
import time
import torch

# --- WCT Helper Functions (from wct.py) ---
# The core WCT logic to be applied in the Lab color space.

def whiten_and_color(cF, sF):
    """Apply WCT transformation to feature tensors"""
    # Ensure tensors are on the same device
    device = cF.device
    sF = sF.to(device)

    cF = cF.squeeze(0)
    sF = sF.squeeze(0)

    # Flatten spatial dimensions
    c_mean = cF.mean(dim=(1, 2), keepdim=True)
    cF_flat = (cF - c_mean).view(cF.shape[0], -1)

    # Whitening
    c_cov = cF_flat @ cF_flat.t() / (cF_flat.shape[1] - 1)
    # Use torch.linalg.svd for newer PyTorch versions
    try:
        c_u, c_e, _ = torch.linalg.svd(c_cov)
    except AttributeError:
        c_u, c_e, _ = torch.svd(c_cov) # Fallback for older PyTorch
        
    k_c = (c_e > 1e-5).sum()
    c_d = torch.diag(c_e[:k_c].pow(-0.5))
    whitened = c_u[:, :k_c] @ c_d @ c_u[:, :k_c].t() @ cF_flat

    # Style
    s_mean = sF.mean(dim=(1, 2), keepdim=True)
    sF_flat = (sF - s_mean).view(sF.shape[0], -1)
    s_cov = sF_flat @ sF_flat.t() / (sF_flat.shape[1] - 1)
    
    try:
        s_u, s_e, _ = torch.linalg.svd(s_cov)
    except AttributeError:
        s_u, s_e, _ = torch.svd(s_cov) # Fallback
        
    k_s = (s_e > 1e-5).sum()
    s_d = torch.diag(s_e[:k_s].pow(0.5))
    colored = s_u[:, :k_s] @ s_d @ s_u[:, :k_s].t() @ whitened

    # Reshape and restore style mean
    out = colored.view_as(cF) + s_mean
    return out.unsqueeze(0)


class BatchWCTProcessor:
    def __init__(self, device=None):
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")

    # --- Color Space Conversion ---
    def rgb_to_lab(self, img):
        """Convert RGB image (uint8) to normalized Lab (float32)"""
        img_norm = img.astype(np.float32) / 255.0
        lab = cv2.cvtColor(img_norm, cv2.COLOR_RGB2LAB)
        # Normalize Lab values: L: [0,100], a,b: [-127,127] -> [0,1]
        lab[:,:,0] = lab[:,:,0] / 100.0
        lab[:,:,1] = (lab[:,:,1] + 128.0) / 255.0
        lab[:,:,2] = (lab[:,:,2] + 128.0) / 255.0
        return lab

    def lab_to_rgb(self, lab):
        """Convert normalized Lab (float32) to RGB image (uint8)"""
        lab_denorm = lab.copy().astype(np.float32)
        lab_denorm[:,:,0] = lab_denorm[:,:,0] * 100.0
        lab_denorm[:,:,1] = lab_denorm[:,:,1] * 255.0 - 128.0
        lab_denorm[:,:,2] = lab_denorm[:,:,2] * 255.0 - 128.0
        rgb = cv2.cvtColor(lab_denorm, cv2.COLOR_LAB2RGB)
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return rgb

    # --- MODIFIED: Core WCT Stylization Function (Luminance Aware) ---
    def wct_lab_transfer(self, content_img, style_img, alpha=1.0):
        """
        Apply WCT in Lab color space to COLOR CHANNELS ONLY, preserving content luminance.
        alpha: Blending strength. 1.0 for full style, 0.0 for original content.
        """
        # Resize style to match content dimensions
        h, w, _ = content_img.shape
        style_img_resized = cv2.resize(style_img, (w, h), interpolation=cv2.INTER_CUBIC)

        # Convert images to normalized Lab space
        content_lab = self.rgb_to_lab(content_img)
        style_lab = self.rgb_to_lab(style_img_resized)

        # --- LUMINANCE PRESERVATION ---
        # Separate the content's L channel and the color (a,b) channels for both images
        content_L = content_lab[:, :, 0]
        content_ab = content_lab[:, :, 1:]
        style_ab = style_lab[:, :, 1:]

        # Convert numpy a,b channels to PyTorch tensors. These are 2-channel tensors.
        content_ab_tensor = torch.from_numpy(content_ab).permute(2, 0, 1).unsqueeze(0).to(self.device)
        style_ab_tensor = torch.from_numpy(style_ab).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Apply WCT to the 'a' and 'b' color channels only
        with torch.no_grad():
            stylized_ab_tensor = whiten_and_color(content_ab_tensor, style_ab_tensor)

        # Convert the stylized color channels back to a numpy array
        stylized_ab = stylized_ab_tensor.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()

        # Create the final Lab image by combining the ORIGINAL content L channel
        # with the NEW stylized a,b channels.
        stylized_lab = np.zeros_like(content_lab)
        stylized_lab[:, :, 0] = content_L
        stylized_lab[:, :, 1:] = stylized_ab
        # --- END OF LUMINANCE PRESERVATION ---

        # Convert the newly combined Lab image back to RGB
        stylized_rgb = self.lab_to_rgb(stylized_lab)

        # Alpha blending allows for fine-tuning the style intensity
        # It blends the original content image with the luminance-preserved stylized image
        blended_rgb = cv2.addWeighted(content_img, 1 - alpha, stylized_rgb, alpha, 0)

        return blended_rgb

    # --- Batch Processing Framework ---
    def load_image(self, image_path):
        """Load image and convert from BGR to RGB"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None

    def get_image_files(self, directory):
        """Get all image files from a directory"""
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
        return sorted(image_files)

    def process_single_pair(self, source_path, target_path, output_path, alpha=1.0):
        """Process a single source-target image pair"""
        source_img = self.load_image(source_path)
        target_img = self.load_image(target_path)

        if source_img is None or target_img is None:
            return False, "Failed to load images"

        try:
            # Apply WCT stylization
            result_rgb = self.wct_lab_transfer(source_img, target_img, alpha=alpha)

            # Save result (convert back to BGR for OpenCV)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(str(output_path), result_bgr)
            
            return success, "Success" if success else "Failed to save image"

        except Exception as e:
            return False, f"Processing error: {str(e)}"

    def batch_process(self, source_dir, output_dir, target_dir=None, target_file=None, alpha=1.0):
        """Main batch processing loop"""
        os.makedirs(output_dir, exist_ok=True)
        source_files = self.get_image_files(source_dir)
        
        results = []
        successful, failed = 0, 0
        
        # Determine processing mode
        if target_dir:
            print(f"Processing pairs from '{source_dir}' and '{target_dir}'...")
            iterator = tqdm(source_files, desc="Processing pairs")
        elif target_file:
            print(f"Processing all sources in '{source_dir}' with single target '{target_file}'...")
            iterator = tqdm(source_files, desc="Processing with single target")
        else:
            return

        for source_path in iterator:
            source_name = Path(source_path).stem
            target_path = None

            if target_dir: # Paired mode
                # Find corresponding target file
                for ext in self.supported_formats:
                    potential_target = os.path.join(target_dir, f"{source_name}{ext}")
                    if os.path.exists(potential_target):
                        target_path = potential_target
                        break
            elif target_file: # Single target mode
                target_path = target_file

            if not target_path or not os.path.exists(target_path):
                status = f"No matching target found for {source_name}"
                print(status)
                failed += 1
                results.append({'source': source_path, 'target': None, 'status': status})
                continue
            
            output_name = f"{source_name}_wct_lab_lum_aware.png" # Recommend PNG for quality
            output_path = os.path.join(output_dir, output_name)

            start_time = time.time()
            success, message = self.process_single_pair(source_path, target_path, output_path, alpha)
            proc_time = time.time() - start_time
            
            if success:
                successful += 1
            else:
                failed += 1
                print(f"Failed to process {source_name}: {message}")
            
            results.append({
                'source': source_path,
                'target': target_path,
                'output': output_path if success else None,
                'status': message,
                'processing_time': proc_time
            })
        
        # Save processing report
        report = {
            'total_processed': len(source_files),
            'successful': successful,
            'failed': failed,
            'settings': {
                'source_dir': source_dir,
                'target_dir': target_dir,
                'target_file': target_file,
                'output_dir': output_dir,
                'alpha': alpha
            },
            'results': results
        }
        report_path = os.path.join(output_dir, 'processing_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)

        print(f"\nProcessing complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Batch Style Transfer using WCT in Lab space (Luminance Aware).')
    
    # Required arguments
    parser.add_argument('--source_dir', required=True, help='Directory containing source images.')
    parser.add_argument('--output_dir', required=True, help='Directory to save processed images.')
    
    # Target specification (mutually exclusive)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--target_dir', help='Directory with corresponding target images (for pair processing).')
    target_group.add_argument('--target_file', help='Path to a single target image (for single target processing).')
    
    # Optional arguments
    parser.add_argument('--alpha', type=float, default=1.0, help='Stylization strength (0.0 to 1.0). 1.0 is full style. Default: 1.0.')
    parser.add_argument('--device', type=str, default=None, help='Device to run PyTorch on (e.g., "cpu", "cuda"). Defaults to auto-detect.')

    args = parser.parse_args()
    
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0.0 and 1.0.")

    # Initialize and run processor
    processor = BatchWCTProcessor(device=args.device)
    processor.batch_process(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        target_dir=args.target_dir,
        target_file=args.target_file,
        alpha=args.alpha
    )

if __name__ == "__main__":
    main()