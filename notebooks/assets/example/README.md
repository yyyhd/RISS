# Example image

`sample_original.png` is a raw, single-channel low-energy (LE) mammography-style
image used only to demonstrate the notebook workflow. It is **not** in the aligned
A|B format expected by the model; the notebook preprocesses it on the fly
(see step 3 of `RISS_inference_demo.ipynb`) following `preprocess/png_png.py`:

1. convert to grayscale and rescale intensities to 0-255;
2. place the LE image in the red channel;
3. append a zero placeholder as the target (right) half;
4. concatenate `[LE | placeholder]` horizontally into the aligned A|B image.

This is a placeholder for demonstration only and should not be used for
quantitative evaluation. For manuscript-level evaluation, replace it with your
own properly preprocessed CEM low-energy or mammography images and run the full
test set.
