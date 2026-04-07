<div align="center">

# Hybrid Forcing
### Long-Horizon Streaming Video Generation via Hybrid Attention with Decoupled Distillation

<p class="subtitle">
<a href="" target="_blank">Ruibin Li</a><sup>1,2</sup>, <a href="" target="_blank">Tao Yang</a><sup>1</sup>, <a href="" target="_blank">Fangzhou Ai</a><sup>1</sup>, <a href="" target="_blank">Tianhe Wu</a><sup>3</sup>, <a href="" target="_blank">Shilei Wen</a><sup>1</sup>, <a href="" target="_blank">Bingyue Peng</a><sup>1</sup>, <a href="" target="_blank">Lei Zhang</a><sup>2</sup><br>
<sup>1</sup>ByteDance &nbsp; <sup>2</sup>The Hong Kong Polytechnic University</a> &nbsp; <sup>3</sup>City University of Hong Kong</a>
</p>

</div>
  </p>
  <h3 align="center"><a href="xxx">Paper</a> | <a href="https://leeruibin.github.io/HybridForcing.github.io/">Website</a> | <a href="xxx">Models</a>  </h3>
</p>



-----


Hybrid Forcing consistently achieves state-of-the-art performance, and thanks to the hybrid attention design the our model achieves real-time video generation at **29.5 FPS (832x480)** on a single NVIDIA H100 GPU without quantization or model compression.


-----



https://github.com/user-attachments/assets/8fe63dae-e303-4ec0-83a4-b53c46f49c09


## 🔥 News
- **2026.4.6** : Work in progress, adapt hybrid video-audio joint streaming generation baseline on LTX2/2.3.
- **2026.4.6** : The [paper](xxx), [project page](https://leeruibin.github.io/HybridForcing.github.io/), and code are released.


## Quick Start

### Installation
```bash
conda create -n hybrid_forcing python=3.10 -y
conda activate hybrid_forcing
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install flash-attn --no-build-isolation

# for hopper gpu, try FA3 and use CUDA 12.8 for best performance
git clone https://github.com/Dao-AILab/flash-attention.git
cd hopper
python setup.py install
```
### Download Checkpoints
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B  --local-dir wan_models/Wan2.1-T2V-1.3B
hf download Wan-AI/Wan2.1-T2V-14B  --local-dir wan_models/Wan2.1-T2V-14B
hf download xxx xxx --local-dir checkpoints
```


### Inference
```bash
python inference.py \
  --config_path configs/hybrid_forcing.yaml \
  --output_folder output/hybrid_forcing \
  --checkpoint_path  xxx \
  --data_path prompts/demos.txt \
```

## Training


### Stage 1: Self-forcing training.

This stage is compatible with Self Forcing training, so you can migrate seamlessly by using the self-forcing code to train an initial stage1 checkpoint for 1000 step.

### Stage 2: Hybrid-Forcing training.

> Set your wandb configs before training.


And then train DMD models:
```bash
  export WANDB_API_KEY=xxx
  
  torchrun --nnodes=8 \
        --nproc_per_node=8 \
        --master_port=xxx \
        --nnodes=xxx \
        --node_rank=xxx \
        train.py \
        --config_path configs/hybrid_forcing.yaml \
        --logdir logs/hybrid_forcing
```
> We recommend training another 1000 steps.

## Acknowledgements
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid), [Self Forcing](https://github.com/guandeh17/Self-Forcing), [Rolling Forcing](https://github.com/TencentARC/RollingForcing), [SLA](https://github.com/thu-ml/SLA), and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo, thanks for their excellect work.

## References
If you find the method useful, please cite
```
@article{li2026long,
  title={Long-Horizon Streaming Video Generation via Hybrid Attention with Decoupled Distillation},
  author={Li, Ruibin and Yang, tao and Ai, Fangzhou and Wu, Tianhe and Wen, shilei and Peng, Bingyue and Zhang, Lei},
  journal={arXiv preprint arXiv:xxx},
  year={2026}
}
```