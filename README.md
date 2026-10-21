Tiny version of [SCRFD](https://arxiv.org/abs/2105.04714) Face and 5 Point Landmark Detection implementation using PyTorch

**English** | [简体中文](README.zh-CN.md)

### Installation

```
conda create -n PyTorch python=3.11.11
conda activate PyTorch
conda install python=3.11.11 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python
pip install PyYAML
pip install tqdm
pip install thop
```

### Docker training

A full Docker-based workflow (image build and acceptance, RetinaFace-WIDER data preparation, 320 finetuning, evaluation, fixed-shape export for ESP-PPQ) is documented in the [Docker training manual](SCRFD-tiny-Docker%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.md) (Chinese), with the image defined by [Dockerfile.scrfd-tiny](Dockerfile.scrfd-tiny).

### Train

* Dataset root is set with `--data-dir` (default `../Dataset/WIDERFace`); checkpoints and `step.csv` go to `--save-dir` (default `./weights`) - point it elsewhere to keep the bundled baseline untouched. Finetune from a checkpoint with `--weights <ckpt.pt>`. Note the defaults are a from-scratch recipe (`--epochs 1200`, `max_lr 0.01` in `utils/args.yaml`): for finetuning use far fewer epochs and consider lowering `max_lr`
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Run `python main.py --test` for testing (loads `best.pt` from `--save-dir`)
* Run `python main.py --export` to export `best.pt` from `--save-dir` as fixed-shape `[1, 3, S, S]` TorchScript + ONNX with no dynamic axes, for ESP-PPQ (S = `--input-size`)

### Results

Upstream numbers at 640 input. This fork defaults to 320 (`--input-size`, anchor sizes auto-scale); expect lower AP at 320, especially on small faces - compare against your own 320 baseline, not this table.

| Model              | AP-easy | AP-medium | AP-hard | Params (M) | Params Ratio | FLOPS (M) | CPU ONNX Latency (ms) | 
|--------------------|---------|-----------|---------|------------|--------------|-----------|-----------------------|
| SCRFD0.5(ICLR2022) | 90.57   | 88.12     | 68.51   | 0.57       | 1.00x        | 508       | 5.5                   | 
| SCRFD-tiny(ours)   | 89.23   | 87.60     | 67.15   | 0.08       | 7.12x        | 451       | 4.4                   | 

### Dataset structure

    ├── WIDERFace 
        ├── images
            ├── train
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train
                ├── 1111.txt
                ├── 2222.txt
            ├── val
                ├── 1111.txt
                ├── 2222.txt

#### Reference

* https://github.com/jahongir7174/YUNet-eval
* https://github.com/ShiqiYu/libfacedetection.train
* https://github.com/deepinsight/insightface/tree/master/detection/scrfd

### License

MIT (see [LICENSE](LICENSE)). Note: the upstream repository (jahongir7174/SCRFD-tiny) ships no license file; this license covers this fork and its modifications.
