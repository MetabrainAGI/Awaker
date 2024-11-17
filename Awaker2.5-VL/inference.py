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
print(output_text)