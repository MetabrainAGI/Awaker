# Awaker

**Awaker** is a series of multimodal large models developed by [Metabrain AGI](https://www.metabrainagi.com)，including multimodal large language model (MLLM) **Awaker-VL**, multimodal retrieval model **Awaker-Sou**, and video generation model **Awaker-Gen**.

## News
- **2024.11.19**: We have released our paper: [Awaker2.5-VL](https://arxiv.org/abs/2411.10669).
- **2024.11.17**: We have released the **Awaker2.5-VL** model. We choose to scale the base MLLM model (like Qwen2-VL-7B) with mixture of experts in a stable and efficient way. This thus leads to the new state-of-the-arts on MME-Realworld and MMBench among all the efficient MLLMs (parameters<30B). The model weights and the inference code of Awaker2.5-VL are now available. Superior open-source Awaker-VL models are coming soon.

## Performance

### MME-RealWorld-CN Benchmark

| Models               | Parameters |      Institutions      | Overall  | Perception | Reasoning |
| ------------------- | :--------: | :--------------------: | :------: | :--------: | :-------: |
| **Awaker2.5-VL (ours)**   |   10.8B    |     Metabrain AGI      | **62.7** | **67.71**  | **52.07** |
| Qwen2-VL            |     8B     |        Alibaba         |   55.5   |   59.80    |   46.46   |
| InternVL-2          |     7B     |    Shanghai AI Lab     |   54.3   |   57.97    |   46.65   |
| InternVL-Chat-V1.5  |    20B     |    Shanghai AI Lab     |   47.9   |   49.90    |   43.74   |
| Claude 3.5 Sonnet   |     -      |       Anthropic        |   47.0   |   48.25    |   44.31   |
| YI-VL-34B           |    34B     |         01.AI          |   42.0   |   42.45    |   41.16   |
| CogVLM2-llama3-Chat |     8B     |     THU & Zhipu AI     |   39.8   |   38.57    |   42.25   |
| GPT-4o              |     -      |         OpenAI         |   38.8   |   43.44    |   29.05   |
| Mini-Gemini-34B-HD  |    34B     |          CUHK          |   38.5   |   38.31    |   38.75   |
| Cambrian-1-8B       |     8B     |          NYU           |   33.6   |   32.44    |   35.97   |
| LLaVA-NeXT-Qwen-72B |    72B     | Bytedance |   30.6   |   30.02    |   31.67   |
| Gemini-1.5-Pro      |     -      |         Google         |   28.1   |   36.10    |   11.14   |
| DeepSeek-VL         |     7B     |      DeepSeek-AI       |   27.6   |   27.63    |   27.63   |
| GPT-4o-mini         |     -      |         OpenAI         |   25.9   |   26.32    |   25.16   |


### MME-RealWorld Benchmark

| Models              | Parameters |     Institutions      | Overall  | Perception | Reasoning |
| ------------------- |:----------:|:---------------------:|:--------:|:----------:|:---------:|
| **Awaker2.5-VL (ours)** |   10.8B    |     Metabrain AGI     | **60.8** | **63.14**  |   43.74   |
| LLaVA-OneVision     |     8B     |       Bytedance       |   57.4   |   59.59    |   41.17   |
| Qwen2-VL            |     8B     |        Alibaba        |   56.5   |   58.96    |   40.39   |
| InternVL-2          |     7B     |    Shanghai AI Lab    |   53.5   |   55.82    |   38.74   |
| Claude 3.5 Sonnet   |     -      |       Anthropic       |   51.6   |   52.90    | **44.12** |
| InternVL-Chat-V1.5  |    20B     |    Shanghai AI Lab    |   49.4   |   51.36    |   36.48   |
| Mini-Gemini-34B-HD  |    34B     |         CUHK          |   45.9   |   48.05    |   31.73   |
| GPT-4o              |     -      |        OpenAI         |   45.2   |   46.43    |   37.61   |
| CogVLM2-llama3-Chat |     8B     |    THU & Zhipu AI     |   44.6   |   45.84    |   37.25   |
| Cambrian-1-8B       |     8B     |          NYU          |   42.7   |   43.82    |   36.16   |
| Gemini-1.5-Pro      |     -      |        Google         |   38.2   |   39.63    |   29.19   |
| GPT-4o-mini         |     -      |        OpenAI         |   36.4   |   37.12    |   32.48   |
| DeepSeek-VL         |     7B     |      DeepSeek-AI      |   32.4   |   33.14    |   27.98   |
| YI-VL-34B           |    34B     |         01.AI         |   31.0   |   30.97    |   32.45   |
| LLaVA-NeXT-Qwen-72B |    72B     | Bytedance |   28.7   |   29.01    |   27.86   |

### MMBench-CN Benchmark

| Models              | Parameters |     Institutions      | Overall  | MMBench_v1.1 | MMBench |
| ------------------- |:----------:|:---------------------:|:--------:|:----------:|:---------:|
|Qwen2-VL-72B | 73.4B | Alibaba | **86.3**  | **85.8**  | **86.7** | 
|InternVL2-40B | 40B | Shanghai AI Lab | 85.7  | 84.9  | 86.4| 
|InternVL2-Llama-76B | 76B | Shanghai AI Lab | 85.5  | 85.5  | -|
|Taiyi | - | Megvii | 85.2  | 85.0  | 85.4 |
|JT-VL-Chat-V3.0 | - | China Mobile | 84.7  | 83.5  | 85.8 |
|LLaVA-OneVision-72B | 73B | ByteDance | 84.6  | 83.9  | 85.3 |
|Step-1.5V | - | StepFun | 84.0  | 83.5  | 84.5 |
|Claude3.5-Sonnet-20241022 | - | Anthropic | 83.0  | 82.5  | 83.5 |
|**Awaker2.5-VL (ours)** | 10.8B | Metabrain AGI | 82.6  | 81.8  | 83.4 |
|GPT-4o (0513, detail-low) | - | OpenAI | 82.3  | 82.5  | 82.1 |
|LLaVA-OneVision-7B | 8B | ByteDance | 81.8  | 80.9  | 82.7 |
|GPT-4o (0513, detail-high) | - | OpenAI | 81.8 | 81.5 | 82.1 |
|InternVL2-26B | 26B | Shanghai AI Lab | 81.5  | 80.9  | 82.1 |
|CongROng | - | CloudWalk | 81.2  | 80.4  | 81.9 |
|MMAlaya2 | 26B | DataCanvas | 80.9  | 79.7  | 82.1 |
|Ovis1.6-Gemma2-9B | 10.2B | Alibaba | 80.8  | 79.5  | 82.0 |
|Qwen2-VL-7B | 8B | Alibaba | 80.5  | 80.3  | 80.6 |
|LLaVA-OneVision-72B (SI)	| 73B | ByteDance | 80.0 | 81.9 | 78.0 |
|InternVL-Chat-V1.5 | 26B | Shanghai AI Lab | 79.9 | 79.1 | 80.7 |
|InternLM-XComposer2.5 | 8B | Shanghai AI Lab | 79.9 | 78.8 | 80.9 |
|GPT-4o (0806, detail-high) | - | OpenAI | 79.8  | 79.2 | 80.3 |
|GPT-4V (0409, detail-high) | - | OpenAI | 79.2  | 78.2 | 80.2 |

### MMBench Benchmark

| Models              | Parameters |     Institutions      | Overall  | MMBench_v1.1 | MMBench |
| ------------------- |:----------:|:---------------------:|:--------:|:----------:|:---------:|
|Qwen2-VL-72B  | 73.4B  | Alibaba  | **86.5**   | **86.1**  | **86.9** |
|InternVL2-40B  | 40B  | Shanghai AI Lab  | 86.0   | 85.1   | 86.8 |
|Taiyi  |  - | Megvii  | 85.7   | 84.7   | 86.7 |
|InternVL2-Llama-76B  | 76B  | Shanghai AI Lab  | 85.5   | 85.5   | - |
|LLaVA-OneVision-72B  | 73B  | ByteDance  | 85.4   | 85.0   | 85.8 |
|JT-VL-Chat-V3.0  | -  | China Mobile  | 84.5   | 83.6   | 85.4 |
|**Awaker2.5-VL (ours)** | 10.8B  | Metabrain AGI  | 83.7   | 82.5   | 84.9 |
|GPT-4o (0513, detail-high) | - | OpenAI | 83.2 | 83.0 | 83.4 |
|GPT-4o (0513, detail-low)  |  - | OpenAI  | 83.2   | 83.1   | 83.3 |
|Step-1.5V  | -  | StepFun  | 82.9   | 80.4   | 85.3 |
|InternVL2-26B  | 26B  | Shanghai AI Lab  | 82.5   | 81.5   | 83.4 |
|Ovis1.6-Gemma2-9B  | 10.2B  | Alibaba  | 82.5   | 81.5   | 83.4 |
|RBDash-v1.2-72B  | 79B  | DLUT  | 82.5   | 81.7   | 83.2 |
|Qwen2-VL-7B  | 8B  | Alibaba  | 82.4   | 81.8   | 83.0 |
|LLaVA-OneVision-7B  | 8B  | ByteDance  | 82.1   | 80.9   | 83.2 |
|GPT-4o (0806, detail-high) |  - | OpenAI  | 82.0   | 81.8   | 82.1 |
|LLaVA-OneVision-72B (SI) | 73B | ByteDance | 81.9 | 83.3 | 80.5 |
|Qwen-VL-Plus-0809  |  - | Alibaba  | 81.9   | 81.1   | 82.7 | 
|CongROng  |  - | CloudWalk  | 81.9   | 80.9   | 82.8 |
|Claude3.5-Sonnet-20241022  | - | Anthropic  | 81.8   | 80.9   | 82.6 |
|MMAlaya2  | 26B  | DataCanvas  | 81.6   | 80.6   | 82.5 |
|InternVL-Chat-V1.5  | 26B  | Shanghai AI Lab  | 81.3 | 80.3 | 82.3 |
|InternLM-XComposer2.5  | 8B  | Shanghai AI Lab  | 81.1 | 80.1 | 82.0 |
|GPT-4V (0409, detail-high)  |  - | OpenAI  | 80.5   | 80.0 | 81.0 |

### MMMU-Pro Benchmark
| Models                  | Parameters | Institutions    | Overall | Vision | Standard |
| ----------------------- | ---------- | --------------- | ------- | ------ | -------- |
| Claude 3.5 Sonnet       | -          | Anthropic       | 51.5    | 48.0   | 55.0     |
| EVLM-KTO                | 7B         | UCF             | -       | 43.8   | -        |
| GPT-4o (0513)           | -          | OpenAI          | 51.9    | 49.7   | 54.0     |
| InternVL2.5-78B         | 78B        | Shanghai AI Lab | 48.6    | 45.9   | 51.4     |
| Gemini 1.5 Pro (0801)   | -          | Google          | 46.9    | 44.4   | 49.4     |
| Qwen2-VL-72B            | 72B        | Alibaba         | 46.2    | 43.3   | 49.2     |
| InternVL2.5-38B         | 38B        | Shanghai AI Lab | 46.0    | 44.0   | 48.0     |
| Gemini 1.5 Pro (0523)   | -          | Google          | 43.5    | 40.5   | 46.5     |
| InternVL2-Llama3-76B    | 76B        | Shanghai AI Lab | 40.0    | 38.0   | 41.9     |
| Llama 3.2 90B           | 90B        |                 | 39.5    | 33.8   | 45.2     |
| GPT-4o mini             | -          | OpenAI          | 37.6    | 35.2   | 39.9     |
| InternVL2.5-26B         | 26B        | Shanghai AI Lab | 37.1    | 32.6   | 41.6     |
| InternVL2.5-8B          | 8B         | Shanghai AI Lab | 34.3    | 30.4   | 38.2     |
| InternVL2-40B           | 40B        | Shanghai AI Lab | 34.2    | 32.1   | 36.3     |
| NVILA                   | 15B        |                 | 33.7    | 28.6   | 38.7     |
| **Awaker2.5-VL (ours)** | 10.8B      | Metabrain AGI   | 31.5    | 24.5   | 38.4     |
| LLaVA-OneVision-72B     | 72B        | ByteDance       | 31.0    | 24.0   | 38.0     |
| Qwen2-VL-7B             | 7B         | Alibaba         | 30.6    | 27.0   | 34.1     |

### MathVista Benchmark
| Models                                   | All  | FQA  | GPS  | MWP  | TQA  | VQA  | ALG  | ARI  | GEO  | LOG  | NUM  | SCI  | STA  |
| ---------------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| OpenAI o1                                | 73.9 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| Pixtral Large（124B）                    | 69.4 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| Grok-2                                   | 69.0 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| Grok-2 mini                              | 68.1 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| Cluaude 3.5                              | 67.7 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| LLaVA-Onevision                          | 67.5 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| InternVL2-8B-MPO                         | 67.3 | 72.5 | 73.6 | 69.9 | 66.5 | 50.3 | 70.1 | 57.5 | 71.5 | 27   | 43.1 | 65.5 | 79.1 |      |
| Ovis1.6-Gemma2-9B                        | 67.2 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| InternVL2-Pro                            | 66.8 | 70.6 | 65.4 | 76.9 | 71.5 | 48   | 66.5 | 62.3 | 63.6 | 27.0 | 40.3 | 65.6 | 81.1 |      |
| Chimera-Reasoner-8B                      | 66.2 | 65.4 | 71.6 | 74.7 | 66.5 | 52.0 | 68.3 | 58.9 | 70.3 | 21.6 | 45.8 | 61.5 | 73.1 |      |
| TextGrad(GPT-4o)                         | 66.1 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
|**Awaker2.5-VL (ours)** | 64.5 | 67.2 | 64.4 | 70.4 | 61.4 | 57   | 61.6 | 59.5 | 63.6 | 16.2 | 43.1 | 63.9 | 74.4 |      |
| Gemini1.5Pro                             | 63.9 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| GPT4o                                    | 63.8 | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    | -    |      |
| Qwen2-VL-7B                              | 60.9 | 70.3 | 43.2 | 67.7 | 62.7 | 58.7 | 47.3 | 59.5 | 44.8 | 27.2 | 41   | 63.9 | 77.7 |      |
| InternVL-Chat-v1.2-plus                  | 59.9 | 51.7 | 61.1 | 79.6 | 52.5 | 57   | 54.5 | 63.2 | 61.1 | 16.2 | 48.6 | 55.7 | 60.8 |      |


### MathVerse Benchmark
| Models                   | Parameters | Institutions    | Overall | Text Domain | Text Lite | Vision Instensive | Vision Domain | Vision Only | Text Only |
| ------------------------ | ---------- | --------------- | ------- | ----------- | --------- | ----------------- | ------------- | ----------- | --------- |
| GPT-4V                   | -          | OpenAI          | 39.40   | 54.7        | 41.4      | 34.9              | 34.4          | 31.6        | 48.7      |
| **Awaker2.5-VL(ours)**   | 10.8B      | Metabrain AGI   | 32.47   | 39.2        | 35.0      | 34.6              | 34.5          | 25.4        | 26.1      |
| Qwen2-VL-7B-Instruct     | 8B         | Alibaba         | 29.63   | 34.1        | 31.1      | 31.2              | 32.4          | 25.0        | 24.0      |
| Qwen-VL-Max              | -          | Alibaba         | 25.88   | 30.7        | 26.1      | 24.1              | 24.1          | 21.4        | 28.9      |
| Gemini-Pro               | -          | Google          | 24.10   | 26.3        | 23.5      | 23.0              | 22.3          | 22.2        | 27.3      |
| LLaVA-NeXT-34B           | 35B        | ByteDance       | 23.35   | 33.8        | 25.5      | 23.5              | 20.3          | 15.7        | 21.3      |
| G-LLaVA-13B              | 14B        | Huawei\         | HKU     | 17.98       | 21.2      | 20.9              | 17.7          | 14.8        | 10.0      |
| G-LLaVA-7B               | 8B         | Huawei\         | HKU     | 17.32       | 20.9      | 20.7              | 17.2          | 14.6        | 9.4       |
| InternLM-Xcomposer-VL-7B | 8B         | Shanghai AI Lab | 16.48   | 22.3        | 17.0      | 15.7              | 16.4          | 11.0        | 16.5      |
| LLaVA-NeXT-13B           | 14B        | ByteDance       | 16.07   | 19.4        | 15.2      | 16.8              | 15.2          | 11.3        | 18.5      |
| SPHINX-MoE               | -          | Shanghai AI Lab | 15.57   | 22.2        | 16.4      | 14.8              | 12.6          | 9.1         | 18.3      |
| SPHINX-Plus              | -          | Shanghai AI Lab | 12.65   | 13.9        | 11.6      | 11.6              | 13.5          | 10.4        | 14.9      |
| Qwen-VL-Plus             | -          | Alibaba         | 12.22   | 15.7        | 11.1      | 9.0               | 13.0          | 10.0        | 14.5      |
| ShareGPT4V-13B           | 13B        | USTC            | 11.97   | 16.2        | 16.2      | 15.3              | 13.8          | 3.7         | 6.6       |



## Environment Requirements

1. Clone this repository and navigate to ```Awaker``` folder.

```bash
git clone https://github.com/MetabrainAGI/Awaker.git
cd Awaker/Awaker2.5-VL
```

2. Install Package.

```bash
# Install specific transformers
cd transformers
pip install -e .
cd ..
# Install specific peft
pip install peft==0.6.0
cp -r peft /path/to/envs/site-packages/
# Install qwen-vl-utils
pip install qwen-vl-utils[decord]
```
3. Version of torch
```bash
torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
```

## Quickstart

You need to download the model weights of Awaker2.5-VL (the ```pytorch_model.bin``` file) from [MetabrainAGI/Awaker2.5-VL](https://huggingface.co/MetabrainAGI/Awaker2.5-VL).


Here we present a code snippet to show how to use the chat model:

```bash
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import MoeConfig, get_peft_model

def find_n_position(target_list, target_value, n):
    count = 0
    for i, element in enumerate(target_list):
        if element == target_value:
            count += 1
            if count == n:
                return i
        
    return -1

# Load the base Qwen2-VL model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# Load the Awaker2.5-VL model
target_modules_for_lora = ["q_proj", "k_proj","v_proj"]
target_modules_for_moe = ["o_proj", "gate_proj", "up_proj", "down_proj"]
num_experts = 4
g_enable = True
lora_config = MoeConfig(
    r=256,
    lora_alpha=512,
    target_modules=target_modules_for_lora,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    modules_to_save=None,
)
moe_config = MoeConfig(
    r=256,
    lora_alpha=512,
    target_modules=target_modules_for_moe,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    modules_to_save=None,
    multiple_loras=True,
    g_enable=g_enable,
    noise_std=0.1,
    gates_tmp=1.0,
    topk=1,
    num_experts=num_experts,
    loss_coef=0,
    token=False,
    freeze_gate=True,
)
model = get_peft_model(model, lora_config, adapter_name='default')
for i in range(num_experts):
    model.add_adapter(str(i), moe_config)
if g_enable:
    model.add_adapter("g", moe_config)
    
# Load the weights of Awaker2.5-VL    
ckpt = torch.load("/path/to/Awaker2.5-VL/pytorch_model.bin")
model.load_state_dict(ckpt, strict=True)
model.to("cuda")
model.eval()

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

vision_start_id = 151652
vision_end_id = 151653
im_start_id = 151644
im_end_id = 151645
prompt_pos = [[0,0]]
input_ids = inputs["input_ids"][0].tolist()
if image_inputs:
    start_pos = input_ids.index(vision_start_id)
else:
    start_pos = find_n_position(input_ids, im_start_id, 2) + 2
end_pos = find_n_position(input_ids, im_end_id, 2)
assert end_pos != -1, "end_pos error!"
assert start_pos != -1,  "start_pos error!"
prompt_pos[0][0] = start_pos
prompt_pos[0][1] = end_pos
inputs["prompt_pos"] = torch.tensor(prompt_pos)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
```

## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.

```
@article{awaker2.5-vl,
    title     = {{Awaker2.5-VL}: Stably Scaling MLLMs with Parameter-Efficient Mixture of Experts},
    author    = {Jinqiang Long and Yanqi Dai and Guoxing Yang and Hongpeng Lin and Nanyi Fei and Yizhao Gao and Zhiwu Lu},    
    journal   = {arXiv preprint arXiv:2411.10669},
    year      = {2024} 
}
```
