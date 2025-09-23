
# Download the pretrained checkpoints:

First, make sure you have installed the huggingface CLI and modelscope CLI.

```bash
pip install -U "huggingface_hub[cli]"
pip install modelscope
```


### Download the pretrained DiT and VAE checkpoints:
```bash
hf download tencent/HunyuanImage-2.1 --local-dir ./ckpts
```

### Downloading TextEncoders

HunyuanImage uses an MLLM and a byT5 as text encoders.

* **MLLM**

    HunyuanImage can be integrated with different MLLMs (including HunyuanMLLM and other open-source MLLM models). 

    At this stage, we have not yet released the latest HunyuanMLLM. We recommend the users in community to use an open-source alternative, such as Qwen2.5-VL-7B-Instruct provided by Qwen Team, which can be downloaded by the following command:
    ```bash
    hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./ckpts/text_encoder/llm
    ```

* **ByT5 encoder**

    We use [Glyph-SDXL-v2](https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2) as our [byT5](https://github.com/google-research/byt5) encoder, which can be downloaded by the following command:

    ```bash
    hf download google/byt5-small --local-dir ./ckpts/text_encoder/byt5-small
    modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./ckpts/text_encoder/Glyph-SDXL-v2
    ```
    You can also manually download the checkpoints from [here](https://modelscope.cn/models/AI-ModelScope/Glyph-SDXL-v2/files) and place them in the text_encoder folder like:
    ```
    ckpts
    ├── text_encoder
    │   ├── Glyph-SDXL-v2
    │   │   ├── assets
    │   │   │   ├── color_idx.json
    │   │   │   ├── multilingual_10-lang_idx.json
    │   │   │   └── ...
    │   │   └── checkpoints
    │   │       ├── byt5_model.pt
    │   │       └── ...
    │   └─  ...
    └─  ...
    ```

* **Reprompt model**

    [PromptEnhancer-32B](https://huggingface.co/PromptEnhancer/PromptEnhancer-32B) can be used as the reprompt model. You can download it by the following command:
    ```bash
    hf download PromptEnhancer/PromptEnhancer-32B --local-dir ./ckpts/reprompt_32b
    ```

<details>

<summary>💡Tips for using hf/huggingface-cli (network problem)</summary>

##### 1. Using HF-Mirror

If you encounter slow download speeds in China, you can try a mirror to speed up the download process:

```shell
HF_ENDPOINT=https://hf-mirror.com hf download tencent/HunyuanImage-2.1 --local-dir ./ckpts
```

##### 2. Resume Download

`huggingface-cli` supports resuming downloads. If the download is interrupted, you can just rerun the download 
command to resume the download process.

Note: If an `No such file or directory: 'ckpts/.huggingface/.gitignore.lock'` like error occurs during the download 
process, you can ignore the error and rerun the download command.

</details>
