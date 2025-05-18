## AI-Based Generation of Synthetic Contrast-Enhanced Mammography Images

Conventional mammography, despite its widespread use, has limited sensitivity and carries a risk of missing lesions. Contrast-enhanced mammography (CEM) offers a more powerful supplement, enhancing the screening and diagnosis of breast lesions and treatment response assessment. However, it requires increased radiation exposure and the use of contrast agents. To address these limitations, we propose a deep learning-based Recombined Image Synthesis System (RISS) that generates high-fidelity recombined images from low-energy images. It integrates the contextual awareness of vision transformers, the precision of convolutional operators, and the realism conferred by adversarial learning, aiming to reduce CEM radiation dose and apply it to mammography to eliminate the need for contrast agents. RISS was trained and tested on a large-scale dataset comprising 113,351 images from 13,335 patients across six hospitals in China, two international public cohorts, and a prospective clinical trial population (Chinese Clinical Trial: ChiCTR2400091510). RISS outperformed classical models and an ablated variant in quantitative evaluations. The system demonstrated superior image fidelity, achieving peak signal-to-noise ratios of 29.71–32.68 and structural similarity indices of 0.90–0.98 (P < 0.05). Radiologists’ evaluations revealed high consistency between real and synthetic recombined images (P > 0.05). In the diagnostic task, synthetic images achieved an area under the receiver operating characteristic curve of ≥ 0.80 , with a kappa coefficient of 0.81 to assess neoadjuvant chemotherapy response. When extended to mammography, RISS eliminated the need for contrast agents and the associated 20–30 minutes of preparation, injection, and observation. This capability significantly enhanced diagnostic sensitivity and reduced unnecessary biopsies (P < 0.05) compared to mammography alone. These findings highlight RISS as a potential tool for improving breast cancer screening and diagnosis while mitigating radiation exposure and contrast-related risks.


## Installation
First clone the repo and cd into the directory:

```bash
git clone https://github.com/yyyhd/RISS
cd RISS
```
Create a new enviroment with anaconda.

## Dependencies
```
python>=3.6.13
torch>=1.10.1
torchvision>=0.8.2
visdom
dominate
scikit-image
h5py
scipy
ml_collections
CUDA >= 11.2
```
### 🔗 Download pre-trained ViT models from Google

https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz &&
```
../model/vit_checkpoint/imagenet21k/
└── R50-ViT-B_16.npz
```


### 🔗 Download Model Weights

You can download the trained weights from [HuggingFace Hub](https://huggingface.co/baguai/RISS1/resolve/main/latest_net_G.pth):

### 🔧 Model Weights Usage

Please place the downloaded model weights file (e.g., `latest_net_G.pth`) in the following directory:
```
/Checkpoints/le_re/ 
├── latest_net_G.pth
```

## Dataset
Before the CEM images were input into the RISS, an algorithm based on automatic threshold segmentation (Otsu) was implemented to isolate the mammary gland from the background. Following the cropping of the breast area, the images were  unified to a resolution of 512×1024 pixels and normalized. To create a two-dimensional image suitable for input into the generator network, the low-energy image and recombined image for each view, along with all-zero images (CC and MLO views), were assigned to the red, green, and blue channels of the RGB image, respectively. During the subsequent network training process, the network was guided to generate a recombined image for the green channel based on the low-energy image in the red channel.
```
/Datasets/
  ├── train
  ├── val
  ├── le_re  ├── test
  ├── mammography_re  ├── test
  ├── public  ├── test
```

## Basic Usage: Generate recombind images from low-energy images:
```
python3 test.py --dataroot Datasets/IXI/dataset/le_re/ --name le_re --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 10000 --serial_batches --fineSize 256 --loadSize 256  --results_dir /IXI/result --checkpoints_dir /IXI/Checkpoints --which_epoch latest
```
## Basic Usage: Generate recombind images from mammography images:
```
python3 test.py --dataroot Datasets/IXI/dataset/mammography_re/ --name le_re --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 10000 --serial_batches --fineSize 256 --loadSize 256  --results_dir /IXI/result --checkpoints_dir /IXI/Checkpoints --which_epoch latest
```
## Evaluation
### Reproducibility:
To reproduce the results in our paper, please download the low-energy and minimized contrast-enhanced mammography image classification digital database [(CDD-CESM)](https://github.com/omar-mohamed/CDD-CESM-Dataset) and contrast-enhanced spectral mammography at Universitario Campus Bio-Medico [CESM@UCBM](http://www.cosbi-lab.it/cesmucbm/). Place the data in the `/dataset/public/test`.

## Basic Usage: Generate recombind images from public images:
```
python3 test.py --dataroot Datasets/IXI/dataset/public/ --name le_re --gpu_ids 0 --model resvit_one --which_model_netG resvit --dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 1 --how_many 10000 --serial_batches --fineSize 256 --loadSize 256  --results_dir /IXI/result --checkpoints_dir /IXI/Checkpoints --which_epoch latest
```
## evaluation indicator
```
cd Evaluation
python PSNR.py
python SSIM.py
```
## Acknowledgements
The project was built on many amazing open-source repositories: ResViT,  pGAN, and pix2pix. We thank the authors and developers for their contributions.

## Issues
Please open new threads or address questions to maoning@pku.edu.cn or sen.yang.scu@gmail.com

## License
This model may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of the RISS model and its derivatives, which include models trained on outputs from the RISS model or datasets created from the RISS model, is prohibited and requires prior approval.

