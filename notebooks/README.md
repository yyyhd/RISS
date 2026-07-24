# RISS notebooks

This directory contains lightweight notebooks for using pretrained RISS models.

- `RISS_inference_demo.ipynb`: step-by-step inference demo. It starts from a raw
  original low-energy mammography image, preprocesses it into the aligned A|B
  format used during training, runs inference with the repository `test.py`
  entry point, and visualizes the synthetic recombined image.

The full workflow is: **raw original image → preprocess → aligned format →
model inference → visualization**.

The bundled example lives in `assets/example/sample_original.png`. The notebook
is intended as an inference walk-through, not as a full reproduction of all
manuscript-level quantitative evaluations.
