# semanticwct-baselines

The classical colour transfer methods I compared before moving on to the neural pipeline in
[`../semanticwct`](../semanticwct). Each one is a standalone script with no shared package, needing
only OpenCV plus NumPy or PyTorch. The original runs used 10 content/style pairs.

## Methods

| File | Method |
|---|---|
| `reinhard.py` | Reinhard. Per-channel mean and std matching in CIELAB, an `alpha` for chroma scaling, side-by-side sheets as output |
| `pca_zca.py` | PCA/ZCA whitening and colouring on RGB pixels. Eigen-decomposition of the 3x3 covariance with epsilon regularisation |
| `adain.py` | AdaIN-style statistics transfer. Normalise the content, rescale with the style mean and std against a contrast-enhanced style, `alpha` control |
| `pixel_wct.py` | WCT on raw RGB pixels. SVD of the 3xN covariance, rank truncation, handles RGBA |
| `lab_wct_luminance.py` | WCT on the Lab a*/b* channels only and the content L* channel kept. Batch CLI, writes a JSON timing report |
| `luminance_blend.py` | Post-processing. Luminance-weighted blending of the content into the stylised image, gamma curve on the content luminance, plus a YUV luminance restore |
| `guided_filter.py` | Post-processing. Guided filter smoothing, to cut the colour bleeding around edges |

## Data

No images are in the repo. `content/` and `styles/` are empty apart from a `.gitkeep` each:

- `content/` is where the content images go. The single-pair scripts read `content/4.jpg`.
- `styles/` holds the matching styles, so `styles/4.jpg` for that one.

The batch scripts glob `content/*.jpg` and look the partner up by filename in `styles/`, so a pair
is `content/<name>.jpg` and `styles/<name>.jpg`. Reinhard writes into `reinhardoutput/`, creating
it, and `lab_wct_luminance.py` writes wherever `--output_dir` points.

## Running

```bash
python reinhard.py                        # batch, writes reinhardoutput/{full,compare}
python adain.py                           # writes adain_{weak,normal,strong}.png
python pca_zca.py                         # writes pca_color_transfer_fixed.png
python pixel_wct.py                       # writes barewct.png

python lab_wct_luminance.py \
    --source_dir content --target_dir styles \
    --output_dir results/lab_wct --alpha 1.0

python luminance_blend.py --content content/4.jpg \
    --stylized results/lab_wct/4_wct_lab_lum_aware.png --output out.jpg
```

## What came out of it

One result file survived, `results/lab_wct/processing_report.json`, the timings from the
luminance-preserving WCT batch. 10 of 10 pairs worked, around 2.5s for the first image and 0.3s
per image after that on CPU.

The grids and per-method images from the original runs are not in the repo, this is code only. For
the record, those grids were the same 10 pairs through Reinhard, pixel-WCT, one simple baseline
and two experimental variants called `hm` and `mk`, each in two colour space settings, direct LAB
transfer and an RGB-to-LAB hybrid. All of it can be regenerated with the scripts above.
