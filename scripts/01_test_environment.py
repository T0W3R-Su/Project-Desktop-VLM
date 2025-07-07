# -- 环境与依赖配置 --
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info

# -- 1. 下载并加载模型与处理器 --

# 从ModelScope社区下载模型文件到本地
print("正在从 ModelScope 下载模型...")
model_dir = snapshot_download('unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit')
print(f"模型已下载至: {model_dir}")

# 加载模型
# Qwen2_5_VLForConditionalGeneration 是Qwen-VL系列的模型类
# bnb 4-bit量化版本，占用显存更少，推理速度更快
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,  # 使用bfloat16以优化性能和显存
    device_map="auto",           # 自动将模型分片加载到可用设备上（如多张GPU）
    attn_implementation="flash_attention_2",  # 使用 Flash Attention 2 加速计算，节省显存
    trust_remote_code=True       # 必须设置为True，因为模型定义在远程代码中
)

# 加载处理器，用于将文本和图像转换为模型可接受的输入格式
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)


# -- 2. 准备输入数据 --

# 构造符合聊天格式的输入消息列表
# 包含一张网络图片和一个文本问题
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "详细描述这张图片。"},
        ],
    }
]

# 将消息列表转换为模型输入的标准格式文本（prompt）
prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# 从消息列表中提取图像URL
image_inputs, video_inputs = process_vision_info(messages)

# 使用处理器对文本和图像进行预处理
# processor会处理图像的下载和转换
inputs = processor(
    text=[prompt],
    images=image_inputs,
    return_tensors="pt"
)

# 将处理好的输入数据移动到模型所在的设备（CPU/GPU）
inputs = inputs.to(model.device)


# -- 3. 模型推理与后处理 --

# 使用.generate()方法进行推理
# max_new_tokens 控制生成文本的最大长度
print("\n开始生成回答...")
generated_ids = model.generate(**inputs, max_new_tokens=512)

# 从生成结果中去除输入的token部分，只保留新生成的内容
# `generated_ids` 包含 "输入" + "回答"，我们需要切片掉 "输入" 部分
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

# 将生成的token ID解码为人类可读的文本
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0] # batch_decode返回一个列表，我们取第一个元素


# -- 4. 输出结果 --
print("\n模型回答:")
print(output_text)