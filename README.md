# FC-GAN — Frequency Conditioned GAN for Remote Sensing Image Super Resolution

Super resolution of remote sensing images using a WGAN-GP framework with frequency separation. The key idea is to condition the adversarial loss on the **high-frequency component** of the image, while the generator is supervised on the low-frequency component via L1 loss, leading to sharper textures without blurry artifacts.

![Visual comparison of super-resolution results](paper/imgs/comparison.png)

*Rows (top to bottom): AID (airport\_29), UC Merced (tenniscourt13), WHU (center\_90). The leftmost "Reference" column highlights a selected region; remaining columns show a zoomed-in view across methods.*

## Architecture

The model follows a two-stage training strategy:

| Stage | Generator loss | Critic |
|-------|---------------|--------|
| 1 — Reconstruction | L1 (pixel) | disabled |
| 2 — Adversarial | L1 + adversarial + perceptual | enabled |

**Generator** — RRDB (Residual-in-Residual Dense Block) network, based on ESRGAN. Upsamples LR input 4× via two PixelShuffle layers.

**Critic** — VGG-style discriminator. When frequency separation is enabled (`freq_sep: true`), the critic operates only on the high-frequency residual.

## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA-capable GPU. Tested with Python 3.11 and PyTorch 2.7.

## Dataset

Place HR images under a directory with the following layout:

```
<data_path>/
  train/         # HR training images
  val/           # HR validation images
  sample_data/   # HR images for visual logging during training
```

LR images are generated on-the-fly from the HR images via 4× bicubic downscaling.

## Configuration

Edit `config.yaml` before running. Key fields:

```yaml
job: 'my_experiment'      # outputs saved to <save_folder>/<job>/
save_folder: './tests'
data_path: './AID'
device: 'cuda'

2nd_stage: true           # stage 1 = false (L1 only), stage 2 = true (+ adversarial + perceptual)
freq_sep: true            # enable frequency conditioning
filter_size: 5            # low-pass kernel size (only used when freq_sep: true)

vgg_layers: {2: 0.1, 7: 0.1, 16: 1.0, 25: 1.0, 34: 1.0}  # {layer_index: weight}
```

## Training

```bash
python train.py -c config.yaml
```

Training resumes automatically from the latest checkpoint if `<save_folder>/<job>/checkpoints/` is not empty. Checkpoints are saved every `save_every` epochs as `G_{epoch:04d}.pth` (and `C_{epoch:04d}.pth` in stage 2).

Monitor with TensorBoard:

```bash
tensorboard --logdir <save_folder>/<job>/logs
```

## Inference

Generate super-resolved images and save them to disk:

```bash
python predict.py -c config.yaml -d <path/to/image/directory>
```

Outputs are written to `<save_folder>/<job>/predictions/`.

