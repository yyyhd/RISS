# RISS Inference Demo

A lightweight, self-contained walk-through for generating a **synthetic recombined image** from a raw low-energy (LE) mammography image using pretrained RISS weights.

The demo covers the full pipeline: **raw original image → preprocessing → aligned format → model inference → visualization**.

<p align="center">
  <img src="notebooks/assets/example/demo_comparison.png" alt="RISS demo: raw low-energy input (left) vs. synthetic recombined output (right)" width="70%">
</p>
<p align="center">
  <em>Left: raw low-energy input &nbsp;•&nbsp; Right: RISS synthetic recombined image</em>
</p>

---

## What the demo does

1. **Start from a raw image.** The bundled example `notebooks/assets/example/sample_original.png` is a single-channel low-energy mammography-style scan. It is **not** yet in the aligned format the model expects.
2. **Preprocess** it into the training-time aligned `A|B` layout (following `preprocess/png_png.py`):
   - convert to grayscale and rescale intensities to `0–255`;
   - place the LE image in the **red** channel;
   - append an all-zero **placeholder** as the target (right) half;
   - concatenate `[LE | placeholder]` horizontally into a single 3-channel PNG.
3. **Run inference** with the repository `test.py` entry point.
4. **Visualize** the raw input, the preprocessed LE, and the synthetic recombined output.

Reproducing this exact preprocessing keeps the input on the training distribution, so the synthetic output stays faithful.

---

## Requirements

- Python 3.x, PyTorch (CUDA recommended; CPU works but is slow)
- Pretrained generator weights at:
  ```text
  Checkpoints/le_re/latest_net_G.pth
  ```

---

## Option A — Run the notebook (recommended)

```bash
jupyter notebook notebooks/RISS_inference_demo.ipynb
```

Run the cells top to bottom. The notebook checks the environment, preprocesses the bundled example, runs inference, and displays a three-panel comparison (raw original → preprocessed LE → synthetic recombined).

## Option B — Run from the command line

```bash
python test.py \
  --dataroot Datasets/notebook_demo \
  --name le_re \
  --gpu_ids 0 \
  --model resvit_one \
  --which_model_netG resvit \
  --pre_trained_transformer 0 \
  --dataset_mode aligned \
  --norm batch \
  --phase test \
  --output_nc 1 \
  --input_nc 3 \
  --how_many 1 \
  --serial_batches \
  --fineSize 256 \
  --loadSize 256 \
  --results_dir results/notebook_demo \
  --checkpoints_dir Checkpoints \
  --which_epoch latest
```

Use `--gpu_ids -1` to run on CPU.

---

## Output

Results are written to:

```text
results/notebook_demo/le_re/test_latest/images/
├── sample_original_aligned_real_A.png   # preprocessed LE input
├── sample_original_aligned_real_B.png   # target placeholder (all zeros)
└── sample_original_aligned_fake_B.png   # ← synthetic recombined image
```

`sample_original_aligned_fake_B.png` is the key result: the recombined image the model synthesizes from the low-energy input.

<p align="center">
  <img src="notebooks/assets/example/sample_recombined.png" alt="Synthetic recombined output" width="30%">
</p>

---

## Use your own data

Place your raw low-energy images through the same preprocessing (see step 3 of the notebook or `preprocess/png_png.py`), write the aligned PNGs under `Datasets/<your_dataset>/test/`, and point `--dataroot` at that directory.

> **Note:** This is a minimal inference demonstration on a placeholder example image. It is **not** intended to reproduce manuscript-level quantitative results, which require the full test cohorts, standardized preprocessing, and the evaluation scripts (PSNR / SSIM) described in the main README.
