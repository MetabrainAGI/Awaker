# Awaker

Awaker is a series of large multimodal models with mixture of LoRA experts developed by MetabrainAGI.

## News

- **2024.10.15**: We have released the Awaker-2.5 model. The model weights and the inference code are now available. Superior open-source Awaker models are coming soon.

## Performance

**MME-RealWorld-CN Benchmark** 

| Methods             | LLM Params | Institutions           | Overall | Perception | Reasoning |
| ------------------- | ---------- | ---------------------- | ------- | ---------- | --------- |
| Awaker-2.5          | 7B+5×0.4B  | Metabrain AGI          | 58.6    | 65.69      | 43.73     |
| Qwen2-VL            | 7B         | Alibaba                | 55.5    | 59.80      | 46.46     |
| InternVL-2          | 7B         | Shanghai AI Lab        | 54.3    | 57.97      | 46.65     |
| InternVL-Chat-V1.5  | 20B        | Shanghai AI Lab        | 47.9    | 49.90      | 43.74     |
| Claude 3.5 Sonnet   | -          | Anthropic              | 47.0    | 48.25      | 44.31     |
| YI-VL-34B           | 34B        | 01.AI                  | 42.0    | 42.45      | 41.16     |
| CogVLM2-llama3-Chat | 8B         | THU & Zhipu AI         | 39.8    | 38.57      | 42.25     |
| GPT-4o              | -          | OpenAI                 | 38.8    | 43.44      | 29.05     |
| Mini-Gemini-34B-HD  | 34B        | CUHK                   | 38.5    | 38.31      | 38.75     |
| Cambrian-1-8B       | 8B         | NYU                    | 33.6    | 32.44      | 35.97     |
| LLaVA-NeXT-Qwen-72B | 72B        | Bytedance & NTU S-Lab  | 30.6    | 30.02      | 31.67     |
| Gemini-1.5-Pro      | -          | Google                 | 28.1    | 36.10      | 11.14     |
| DeepSeek-VL         | 7B         | DeepSeek-AI            | 27.6    | 27.63      | 27.63     |
| GPT-4o-mini         | -          | OpenAI                 | 25.9    | 26.32      | 25.16     |
| ShareGPT4V-13B      | 13B        | USTC & Shanghai AI Lab | 25.9    | 25.75      | 26.17     |


**MME-RealWorld Benchmark**

| Methods             | LLM Params | Institutions           | Overall | Perception | Reasoning |
| ------------------- | ---------- | ---------------------- | ------- | ---------- | --------- |
| Awaker-2.5          | 7B+5×0.4B  | Metabrain AGI          | 58.2    | 60.58      | 40.92     |
| LLaVA-OneVision     | 7B         | Bytedance & NTU S-Lab  | 57.4    | 59.59      | 41.17     |
| Qwen2-VL            | 7B         | Alibaba                | 56.5    | 58.96      | 40.39     |
| InternVL-2          | 7B         | Shanghai AI Lab        | 53.5    | 55.82      | 38.74     |
| Claude 3.5 Sonnet   | -          | Anthropic              | 51.6    | 52.90      | 44.12     |
| InternVL-Chat-V1.5  | 20B        | Shanghai AI Lab        | 49.4    | 51.36      | 36.48     |
| Mini-Gemini-34B-HD  | 34B        | CUHK                   | 45.9    | 48.05      | 31.73     |
| GPT-4o              | -          | OpenAI                 | 45.2    | 46.43      | 37.61     |
| CogVLM2-llama3-Chat | 8B         | THU & Zhipu AI         | 44.6    | 45.84      | 37.25     |
| Cambrian-1-8B       | 8B         | NYU                    | 42.7    | 43.82      | 36.16     |
| Gemini-1.5-Pro      | -          | Google                 | 38.2    | 39.63      | 29.19     |
| GPT-4o-mini         | -          | OpenAI                 | 36.4    | 37.12      | 32.48     |
| DeepSeek-VL         | 7B         | DeepSeek-AI            | 32.4    | 33.14      | 27.98     |
| YI-VL-34B           | 34B        | 01.AI                  | 31.0    | 30.97      | 32.45     |
| LLaVA-NeXT-Qwen-72B | 72B        | Bytedance & NTU S-Lab  | 28.7    | 29.01      | 27.86     |
| ShareGPT4V-13B      | 13B        | USTC & Shanghai AI Lab | 27.8    | 28.38      | 24.63     |

## Environment Requirements

```bash
pip install qwen-vl-utils[decord]
```

## Quickstart of Awaker-2.5

You need to download the model weights of Awaker-2.5 (the ```pytorch_model.bin``` file) from [here](https://huggingface.co/MetabrainAGI/Awaker-2.5/resolve/main/pytorch_model.bin).


Here we show a code snippet to show you how to use the chat model with transformers and qwen_vl_utils:

```bash
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from swift.llm import get_model_tokenizer, ModelType
from peft import MoeConfig, get_peft_model

# Load base Qwen2-VL model
model_type = ModelType.qwen2_vl_7b_instruct
model, tokenizer = get_model_tokenizer(
    model_type, torch.bfloat16, 
    model_kwargs={"device_map": "auto"},
    model_id_or_path="Qwen/Qwen2-VL-7B-Instruct"
)
# Load Awaker 2.5 model
target_modules_1 = ["q_proj", "k_proj","v_proj"]
target_modules_2 = ["o_proj", "gate_proj", "up_proj", "down_proj"]
num_experts = 4
g_enable = True
moe_config_1 = MoeConfig(
    r=64,
    lora_alpha=16,
    target_modules=target_modules_1,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    modules_to_save=None,
    multiple_loras=True,
    multiple_mlps=False,
    g_enable=g_enable,
    noise_std=0.1,
    gates_tmp=1.0,
    topk=1,
    num_experts=num_experts,
    loss_coef=0,
    token=False,
    freeze_gate=True,
)
moe_config_2 = MoeConfig(
    r=64,
    lora_alpha=16,
    target_modules=target_modules_2,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    modules_to_save=None,
    multiple_loras=True,
    multiple_mlps=False,
    g_enable=g_enable,
    noise_std=0.1,
    gates_tmp=1.0,
    topk=1,
    num_experts=num_experts,
    loss_coef=0,
    token=False,
    freeze_gate=True,
)
model = get_peft_model(model, moe_config_1, adapter_name='default')
for i in range(num_experts):
    model.add_adapter(str(i), moe_config_2)
if g_enable:
    model.add_adapter("g", moe_config_2)
ckpt = torch.load("PATH_TO_AWAKER_2.5/pytorch_model.bin")
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
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
