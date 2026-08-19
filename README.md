# TT-VLA: On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning

[![Paper](https://img.shields.io/badge/ACL%202026-Paper-blue.svg)](https://aclanthology.org/2026.acl-long.1863/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of **[On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning](https://aclanthology.org/2026.acl-long.1863/)** (ACL 2026).

## Installation

### Environment setup

Clone the repository and run the following commands from its root directory:

```bash
git clone https://github.com/ChangyuLiu2022/TT-VLA.git
cd TT-VLA

conda create -n ttvla  python==3.10
conda activate ttvla

pip install torch==2.2.0 
pip install numpy==1.26.4

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

cd openvla && pip install -e . && cd ..
pip install -U tyro
pip install datasets==3.3.2

# special install for flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# install other dependencies
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..
cd VLAC && pip install -e . && cd ..
pip install tokenizers==0.21.4

#for video flash
pip install av==15.1.0
pip install decord==0.6.0
```



See [`ttvla_env_setup.sh`](ttvla_env_setup.sh) for the optional data-collection environment and additional setup details.

## Model Checkpoints

TT-VLA requires an OpenVLA policy checkpoint and a VLAC reward model. Download both checkpoints before running an experiment:

```bash
huggingface-cli download InternRobotics/VLAC \
  --local-dir /path/to/VLAC

huggingface-cli download gen-robot/openvla-7b-rlvla-warmup \
  --local-dir /path/to/openvla-7b-rlvla-warmup
```

If the Hugging Face CLI is not available, install it with `pip install -U huggingface_hub`.

## Running TT-VLA with OpenVLA

```bash
CUDA_VISIBLE_DEVICES=0,1 \
python SimplerEnv/simpler_env/train_ms3_ppo_ttvla.py \
  --name PPO-v1-warmup \
  --env-id PutOnPlateInScene25MultiPlate-v1 \
  --vla-path /path/to/openvla-7b-rlvla-warmup \
  --vla-unnorm-key bridge_orig \
  --reward-model-path /path/to/VLAC \
  --seed 0 \
  --num-envs 1 \
  --episode-len 160 \
  --alg-name ppo \
  --ttt 1 \
  --tt-steps 8 \
  --max-episodes 80 \
  --no-normalize-advantage \
  --buffer-gamma 0
```

Key arguments:

| Argument | Description |
| --- | --- |
| `--env-id` | Task to run. |
| `--vla-path` | Local path or Hugging Face ID of the OpenVLA checkpoint. |
| `--reward-model-path` | Local path to the VLAC reward model. |
| `--ttt` | Enables (`1`) or disables (`0`) test-time training. |
| `--tt-steps` | Number of environment steps between test-time updates. |
| `--max-episodes` | Number of evaluation episodes. |

The script uses the second visible GPU for the policy when two GPUs are available and the first GPU for VLAC. If only one GPU is visible, both models share that device.

## Task definition

1. `PutOnPlateInScene25VisionImage-v1`-`test`: unseen table
2. `PutOnPlateInScene25VisionTexture03-v1`-`test`: dynamic texture (weak)
3. `PutOnPlateInScene25VisionTexture05-v1`-`test`: dynamic texture (strong)
4. `PutOnPlateInScene25VisionWhole03-v1`-`test`: dynamic noise (weak)
5. `PutOnPlateInScene25VisionWhole05-v1`-`test`: dynamic noise (strong)
6. `PutOnPlateInScene25Carrot-v1`-`train`: similar to training setting
7. `PutOnPlateInScene25Carrot-v1`-`test`: unseen objects
8. `PutOnPlateInScene25Plate-v1`-`test`: unseen receptacles
9. `PutOnPlateInScene25Instruct-v1`-`test`: unseen instructions
10. `PutOnPlateInScene25MultiCarrot-v1`-`train`: multi-object (both seen)
11. `PutOnPlateInScene25MultiCarrot-v1`-`test`: multi-object (both unseen)
12. `PutOnPlateInScene25MultiPlate-v1`-`train`: distractive receptacle
13. `PutOnPlateInScene25MultiPlate-v1`-`test`: multi-receptacle (both unseen)
14. `PutOnPlateInScene25Position-v1`-`test`: unseen position (object & receptacle)
15. `PutOnPlateInScene25EEPose-v1`-`test`: unseen robot init pose
16. `PutOnPlateInScene25PositionChangeTo-v1`-`test`: mid-episode object reposition

## Acknowledgements

This codebase builds on [RL4VLA](https://github.com/gen-robot/RL4VLA) and incorporates components from [OpenVLA](https://github.com/openvla/openvla), [ManiSkill](https://github.com/haosulab/ManiSkill), [SimplerEnv](https://github.com/simpler-env/SimplerEnv), and [VLAC](https://github.com/InternRobotics/VLAC). Refer to RL4VLA for the ManiSkill data-collection pipeline.

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{liu2026fly,
  title     = {On-the-fly vla adaptation via test-time reinforcement learning},
  author    = {Liu, Changyu and Liu, Yiyang and Wang, Taowen and Zhuang, Qiao and Liang, James Chenhao and Yang, Wenhao and Xu, Renjing and Wang, Qifan and Liu, Dongfang and Han, Cheng},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for
               Computational Linguistics (Volume 1: Long Papers)},
  year      = {2026},
  pages     = {40107--40125},
  doi       = {10.18653/v1/2026.acl-long.1863},
  url       = {https://aclanthology.org/2026.acl-long.1863/}
}
```



## License

This repository is released under the [MIT License](LICENSE). Third-party components may be subject to their own licenses.
