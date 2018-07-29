# pytorch-glow
PyTorch implementation of ["Glow: Generative Flow with Invertible 1x1 Convolutions"](https://arxiv.org/abs/1807.03039)

## Usage
First you need to install all requirements by
```shell
pip3 install -r requirements.txt
```

### Training
1. Prepare dataset and corresponding profile file (like `profile/celeba.json`)
2. Start training:
```shell
python3 train.py [profile] 
``` 

### Inference
```
Usage: python3 infer.py [OPTIONS] COMMAND [ARGS]...

Options:
  --profile PATH
  --snapshot PATH
  --help           Show this message and exit.

Commands:
  compute_deltaz
  interpolate
  reconstruct
  sample
```

## Result
### Reconstruction
The upper is reconstructed image, the lower is original one.
![reconstructed result](result/reconstructed.png)

### Interpolation
From left to right, the attribute offset is from `-1.0` to `1.0` by step `0.25`.

- Attractive

![interpolation_attractive](result/interpolated_Attractive.png)

- Black hair

![interpolation_black_hair](result/interpolated_Black_Hair.png)

- Blurry

![interpolation_blurry](result/interpolated_Blurry.png)

- Mouth slightly open

![interpolation_mouth_slightly_open](result/interpolated_Mouth_Slightly_Open.png)

## Acknowledgement
 This project refers to:
  - [openai/glow](https://github.com/openai/glow) (Official implementation)
  - [chaiyujin/glow-pytorch](https://github.com/chaiyujin/glow-pytorch) 
