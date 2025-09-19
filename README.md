
[中文阅读](./README_CN.md)

<p align="center">
  <img src="./assets/logo.png"  height=100>
</p>

<div align="center">

# HunyuanImage-2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation​

</div>


<p align="center"> &nbsp&nbsp🤗 <a href="https://huggingface.co/tencent/HunyuanImage-2.1">HuggingFace</a>&nbsp&nbsp | 
💻 <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=286&from=/visual">Official website(官网) Try our model!</a>&nbsp&nbsp
</p>


<!-- <div align="center">
  <a href=https://github.com/Tencent-Hunyuan/HunyuanImage-2.1 target="_blank"><img src=https://img.shields.io/badge/Code-black.svg?logo=github height=22px></a>
  <a href="https://huggingface.co/spaces/tencent/HunyuanImage-2.1" target="_blank">
    <img src="https://img.shields.io/badge/Demo%20Page-blue" height="22px"></a>
  <a href=https://huggingface.co/tencent/HunyuanImage-2.1 target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href="#" target="_blank"><img src="https://img.shields.io/badge/Report-Coming%20Soon-blue" height="22px"></a><br/>
  <a href="https://www.arxiv.org/abs/2509.04545" target="https://arxiv.org/abs/2509.04545"><img src="https://img.shields.io/badge/PromptEnhancer-Report-yellow" height="22px"></a>
  <a href= https://hunyuan-promptenhancer.github.io/ target="_blank"><img src=https://img.shields.io/badge/PromptEnhancer-bb8a2e.svg?logo=github height=22px></a><br/>
  <a href=https://x.com/TencentHunyuan target="_blank"><img src=https://img.shields.io/badge/Hunyuan-black.svg?logo=x height=22px></a>
</div> -->

<p align="center">
    👏 Join our <a href="assets/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/ehjWMqF5wY">Discord</a>
</p>


-----

This repo contains PyTorch model definitions, pretrained weights and inference/sampling code for our HunyuanImage-2.1. You can <span style="color:red">**directly try our model**</span> on [Official website(官网)](https://hunyuan.tencent.com/modelSquare/home/play?modelId=286&from=/visual) and find more visualizations on our [project page](https://hunyuan.tencent.com/image/en?tabIndex=0).


<div align="center">
  <img src="./assets/demo.jpg" width=100% alt="HunyuanImage 2.1 Demo">
</div>

## 🔥🔥🔥 Latest Updates
- September 18, 2025: ✨ Try the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for higher-quality prompt enhancement!​.
- September 18, 2025: ✨ [ComfyUI workflow of HunyuanImage-2.1](https://github.com/KimbingNg/ComfyUI-HunyuanImage2.1) is available now!
- September 16, 2025: 👑 We achieved the Top1 on Arena's leaderboard for text-to-image open-source models. [Leaderboard](https://artificialanalysis.ai/text-to-image/arena/leaderboard-text)
- September 12, 2025: 🚀 Released FP8 quantized models! Making it possible to generate 2K images with only 24GB GPU memory!
- September 8, 2025: 🚀 Released inference code and model weights for HunyuanImage-2.1.


<!-- ## 🎥 Demo

<div align="center">
  <img src="./assets/demo.jpg" width=100% alt="HunyuanImage 2.1 Demo">
</div> -->

<!-- ## Contents
- [HunyuanImage-2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation​](#hunyuanimage-21-an-efficient-diffusion-model-for-high-resolution-2k-text-to-image-generation)
  - [🔥🔥🔥 Latest Updates](#-latest-updates)
  - [🎥 Demo](#-demo)
  - [Contents](#contents)
  - [Abstract](#abstract)
  - [HunyuanImage-2.1 Overall Pipeline](#hunyuanimage-21-overall-pipeline)
    - [Training Data and Caption](#training-data-and-caption)
    - [Text-to-Image Model Architecture](#text-to-image-model-architecture)
    - [Reinforcement Learning from Human Feedback](#reinforcement-learning-from-human-feedback)
    - [Rewriting Model](#rewriting-model)
    - [Model distillation](#model-distillation)
  - [🎉 HunyuanImage-2.1 Key Features](#-hunyuanimage-21-key-features)
  - [Prompt Enhanced Demo](#prompt-enhanced-demo)
  - [📈 Comparisons](#-comparisons)
    - [SSAE Evaluation](#ssae-evaluation)
    - [GSB Evaluation](#gsb-evaluation)
  - [📜 System Requirements](#-system-requirements)
  - [🛠️ Dependencies and Installation](#️-dependencies-and-installation)
  - [🧱 Download Pretrained Models](#-download-pretrained-models)
  - [🔑 Usage](#-usage)
  - [🔗 BibTeX](#-bibtex)
  - [Acknowledgements](#acknowledgements)
  - [Github Star History](#github-star-history)

--- -->
<!-- - [🧩 Community Contributions](#-community-contributions) -->
## Introduction
We are excited to introduce **HunyuanImage-2.1**, a 17B text-to-image model that is capable of generating **2K (2048 × 2048) resolution** images. 

<!-- Leveraging an extensive dataset and structured captions involving multiple expert models, we significantly enhance text-image alignment capabilities. The model employs a highly expressive VAE with a (32 × 32) spatial compression ratio, substantially reducing computational costs. -->

Our architecture consists of two stages:
1. **​Base text-to-image Model**:​​ The first stage is a text-to-image model that utilizes two text encoders: a multimodal large language model (MLLM) to improve image-text alignment, and a multi-language, character-aware encoder to enhance text rendering across various languages. 
2. **Refiner Model**: The second stage introduces a refiner model that further enhances image quality and clarity, while minimizing artifacts. 
<!-- 
Additionally, we developed the PromptEnhancer module to further boost model performance, and employed meanflow distillation for efficient inference. HunyuanImage-2.1 demonstrates robust semantic alignment and cross-scenario generalization, leading to improved consistency between text and image, enhanced control of scene details, character poses, and expressions, and the ability to generate multiple objects with distinct descriptions. -->

👑 We achieved the **Top1** on Arena's leaderboard for text-to-image open-source models.

<div align="center">
  <img src="./assets/leaderboard.png" width=70% alt="HunyuanImage 2.1 Demo">
</div>
 
<!-- 
## HunyuanImage-2.1 Overall Pipeline

### Training Data and Caption

Structured captions provide hierarchical semantic information at short, medium, long, and extra-long levels, significantly enhancing the model’s responsiveness to complex semantics. Innovatively, an OCR agent and IP RAG are introduced to address the shortcomings of general VLM captioners in dense text and world knowledge descriptions, while a bidirectional verification strategy ensures caption accuracy.


### Text-to-Image Model Architecture

<p align="center">
  <img src="./assets/framework_overall.png" width=100% alt="HunyuanImage 2.1 Architecture">
</p>



Core Components:
* High-Compression VAE with REPA Training Acceleration:
  * A VAE with a 32× compression rate drastically reduces the number of input tokens for the DiT model. By aligning its feature space with DINOv2 features, we facilitate the training of high-compression VAEs. As a result, our model generates 2K images with the same token length (and thus similar inference time) as other models require for 1K images, achieving superior inference efficiency.
  * Multi-bucket, multi-resolution REPA loss aligns DiT features with a high-dimensional semantic feature space, accelerating model convergence.
* Dual Text Encoder:
  * A vision-language multimodal encoder is employed to better understand scene descriptions, character actions, and detailed requirements.
  * A multilingual ByT5 text encoder is introduced to specialize in text generation and multilingual expression.
* Network: A single- and dual-stream diffusion transformer with 17 billion parameters.

### Reinforcement Learning from Human Feedback
Two-Stage Post-Training with Reinforcement Learning: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are applied sequentially in two post-training stages. We introduce a Reward Distribution Alignment algorithm, which innovatively incorporates high-quality images as selected samples to ensure stable and improved reinforcement learning outcomes.

### Rewriting Model
<p align="center">
  <img src="./assets/framework_prompt_rewrite.png" width=90% alt="HunyuanImage 2.1 Architecture">
</p>

* The first systematic industrial-level rewriting model. SFT training structurally rewrites user text instructions to enrich visual expression, while GRPO training employs a fine-grained semantic AlignEvaluator reward model to substantially improve the semantics of images generated from rewritten text. The AlignEvaluator covers 6 major categories and 24 fine-grained assessment points. PromptEnhancer supports both Chinese and English rewriting and demonstrates general applicability in enhancing semantics for both open-source and proprietary text-to-image models.

### Model distillation
We propose a novel distillation method based on meanflow that addresses the key challenges of instability and inefficiency inherent in standard meanflow training. This approach enables high-quality image generation with only a few sampling steps. To our knowledge, this is the first successful application of meanflow to an industrial-scale model.



 -->

## 🎉 HunyuanImage-2.1 Key Features

- **High-Quality Generation**: Efficiently produces ultra-high-definition (2K) images with cinematic composition.
- **Multilingual Support**: Provides native support for both Chinese and English prompts.
- **Advanced Architecture**: Built on a multi-modal, single- and dual-stream combined DiT (Diffusion Transformer) backbone.
- **Glyph-Aware Processing**: Utilizes ByT5's text rendering capabilities for improved text generation accuracy.
- **Flexible Aspect Ratios**: Supports a variety of image aspect ratios (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3).
- **Prompt Enhancement**: Automatically rewrites prompts to improve descriptive accuracy and visual quality.






## 📜 System Requirements

**Hardware and OS Requirements:**
- NVIDIA GPU with CUDA support.

  **Minimum requrement for now:** 24 GB GPU memory for 2048x2048 image generation.
  
  > **Note:** The memory requirements above are measured with model CPU offloading and FP8 quantization enabled. If your GPU has sufficient memory, you may disable offloading for improved inference speed.
- Supported operating system: Linux.


## 🛠️ Dependencies and Installation

1. Clone the repository:
```bash
git clone https://github.com/Tencent-Hunyuan/HunyuanImage-2.1.git
cd HunyuanImage-2.1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

## 🧱 Download Pretrained Models

The details of download pretrained models are shown [here](ckpts/checkpoints-download.md).

## 🔑 Usage

### Prompt Enhancement

Prompt enhancement plays a **crucial role** in enabling our model to generate high-quality images. By writing longer and more detailed prompts, the generated image will be significantly improved. We encourage you to craft comprehensive and descriptive prompts to achieve the best possible image quality. 

We highly recommend you to try the [PromptEnhancer-32B model](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) for higher-quality prompt enhancement.


### Text to Image
HunyuanImage-2.1 **only supports 2K** image generation (e.g. 2048x2048 for 1:1 images, 2560x1536 for 16:9 images, etc.).
Generating images with 1K resolution will result in artifacts.

Additionally, we **highly recommend** using the full generation pipeline for better quality (i.e. enabling prompt enhancement and refinment).


| model type               | model name                | description                             | num_inference_steps | guidance_scale | shift |
|--------------------------|---------------------------|-----------------------------------------|---------------------|----------------|-------|
| Base text-to-image Model | hunyuanimage2.1           | Undistilled model for the best quality. | 50                  | 3.5            | 5     |
| Distilled text-to-image Model | hunyuanimage2.1-distilled | Distilled model for faster inference    | 8                   | 3.25           | 4     |
| Refiner                  | hunyuanimage-refiner      | The refiner model                       | N/A                 | N/A            | N/A   |


```python
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline

# Supported model_name: hunyuanimage-v2.1, hunyuanimage-v2.1-distilled
model_name = "hunyuanimage-v2.1"
pipe = HunyuanImagePipeline.from_pretrained(model_name=model_name, use_fp8=True)
pipe = pipe.to("cuda")

# The input prompt
prompt = "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word “Tencent” on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."


# Generate with different aspect ratios
aspect_ratios = {
    "16:9": (2560, 1536),
    "4:3": (2304, 1792),
    "1:1": (2048, 2048),
    "3:4": (1792, 2304),
    "9:16": (1536, 2560),
}

width, height = aspect_ratios["1:1"]

image = pipe(
    prompt=prompt,
    width=width,
    height=height,
    # disable the reprompt if you already use the prompt enhancement to enhance the prompt
    use_reprompt=False,  # Enable prompt enhancement (which may result in higher GPU memory usage)
    use_refiner=True,   # Enable refiner model
    # For the distilled model, use 8 steps for faster inference.
    # For the non-distilled model, use 50 steps for better quality.
    num_inference_steps=8 if "distilled" in model_name else 50, 
    guidance_scale=3.25 if "distilled" in model_name else 3.5,
    shift=4 if "distilled" in model_name else 5,
    seed=649151,
)

image.save("generated_image.png")
```

## Prompt Enhanced Demo
To improve the quality and detail of generated images, we use a prompt rewriting model. This model automatically enhances user-provided text prompts by adding detailed and descriptive information.
<p align="center">
  <img src="./assets/reprompt.jpg" width=100% alt="Human Evaluation with Other Models">
</p>



## 📈 Comparisons

### SSAE Evaluation
SSAE (Structured Semantic Alignment Evaluation) is an intelligent evaluation metric for image-text alignment based on advanced multimodal large language models (MLLMs). We extracted 3500 key points across 12 categories, then used multimodal large language models to automatically evaluate and score by comparing the generated images with these key points based on the visual content of the images. Mean Image Accuracy represents the image-wise average score across all key points, while Global Accuracy directly calculates the average score across all key points.
<p align="center">
<table>
<thead>
<tr>
    <th rowspan="2">Model</th>  <th rowspan="2">Open Source</th> <th rowspan="2">Mean Image Accuracy</th> <th rowspan="2">Global Accuracy</th> <th colspan="4" style="text-align: center;">Primary Subject</th> <th colspan="3" style="text-align: center;">Secondary Subject</th> <th colspan="2" style="text-align: center;">Scene</th> <th colspan="3" style="text-align: center;">Other</th>
</tr>
<tr>
    <th>Noun</th> <th>Key Attributes</th> <th>Other Attributes</th> <th>Action</th> <th>Noun</th> <th>Attributes</th> <th>Action</th> <th>Noun</th> <th>Attributes</th> <th>Shot</th> <th>Style</th> <th>Composition</th>
</tr>
</thead>
<tbody>
<tr>
    <td>FLUX-dev</td> <td>✅</td> <td>0.7122</td> <td>0.6995</td> <td>0.7965</td> <td>0.7824</td> <td>0.5993</td> <td>0.5777</td> <td>0.7950</td> <td>0.6826</td> <td>0.6923</td> <td>0.8453</td> <td>0.8094</td> <td>0.6452</td> <td>0.7096</td> <td>0.6190</td>
</tr>
<tr>
    <td>Seedream-3.0</td> <td>❌</td> <td>0.8827</td> <td>0.8792</td> <td>0.9490</td> <td>0.9311</td> <td>0.8242</td> <td>0.8177</td> <td>0.9747</td> <td>0.9103</td> <td>0.8400</td> <td>0.9489</td> <td>0.8848</td> <td>0.7582</td> <td>0.8726</td> <td>0.7619</td>
</tr>
<tr>
    <td>Qwen-Image</td> <td>✅</td> <td>0.8854</td> <td>0.8828</td> <td>0.9502</td> <td>0.9231</td> <td>0.8351</td> <td>0.8161</td> <td>0.9938</td> <td>0.9043</td> <td>0.8846</td> <td>0.9613</td> <td>0.8978</td> <td>0.7634</td> <td>0.8548</td> <td>0.8095</td>
</tr>
<tr>
    <td>GPT-Image</td>  <td>❌</td> <td> 0.8952</td> <td>0.8929</td> <td>0.9448</td> <td>0.9289</td> <td>0.8655</td> <td>0.8445</td> <td>0.9494</td> <td>0.9283</td> <td>0.8800</td> <td>0.9432</td> <td>0.9017</td> <td>0.7253</td> <td>0.8582</td> <td>0.7143</td>
</tr>
<tr>
    <td><strong>HunyuanImage 2.1</strong></td> <td>✅</td> <td><strong>0.8888</strong></td> <td><strong>0.8832</strong></td> <td>0.9339</td> <td>0.9341</td> <td>0.8363</td> <td>0.8342</td> <td>0.9627</td> <td>0.8870</td> <td>0.9615</td> <td>0.9448</td> <td>0.9254</td> <td>0.7527</td> <td>0.8689</td> <td>0.7619</td>
</tr>
</tbody>
</table>
</p>

From the SSAE evaluation results, our model has currently achieved the optimal performance among open-source models in terms of semantic alignment, and is very close to the performance of closed-source commercial models (GPT-Image).

### GSB Evaluation

<p align="center">
  <img src="./assets/gsb.png" width=70% alt="Human Evaluation with Other Models">
</p>

We adopted the GSB evaluation method commonly used to assess the relative performance between two models from an overall image perception perspective. In total, we utilized 1000 text prompts, generating an equal number of image samples for all compared models in a single run. For a fair comparison, we conducted inference only once for each prompt, avoiding any cherry-picking of results. When comparing with the baseline methods, we maintained the default settings for all selected models. The evaluation was performed by more than 100 professional evaluators.
From the results, HunyuanImage 2.1 achieved a relative win rate of -1.36% against Seedream3.0 (closed-source) and 2.89% outperforming Qwen-Image (open-source). The GSB evaluation results demonstrate that HunyuanImage 2.1, as an open-source model, has reached a level of image generation quality comparable to closed-source commercial models (Seedream3.0), while showing certain advantages in comparison with similar open-source models (Qwen-Image). This fully validates the technical advancement and practical value of HunyuanImage 2.1 in text-to-image generation tasks.


### Contact
Feel free to join our Discord server or join our WeChat groups—not only to exchange ideas and explore collaboration, but also to ask any questions you might have. You're welcome to open an issue or submit a pull request on GitHub. Your feedback is valuable to us and helps drive HunyuanImage forward. Thank you for being a part of our community!


## 🔗 BibTeX

If you find this project useful for your research and applications, please cite as:

```BibTeX
@misc{HunyuanImage-2.1,
  title={HunyuanImage 2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation},
  author={Tencent Hunyuan Team},
  year={2025},
  howpublished={\url{https://github.com/Tencent-Hunyuan/HunyuanImage-2.1}},
}
```

## Acknowledgements

We would like to thank the following open-source projects and communities for their contributions to open research and exploration: [Qwen](https://huggingface.co/Qwen), [FLUX](https://github.com/black-forest-labs/flux), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co).

## Github Star History
<a href="https://star-history.com/#Tencent-Hunyuan/HunyuanImage-2.1&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Tencent-Hunyuan/HunyuanImage-2.1&type=Date1" />
 </picture>
</a>
