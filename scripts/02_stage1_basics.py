"""
本脚本演示了如何使用 Transformers 和 ModelScope 加载和运行
unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit
旨在保持较高模型性能的同时，显著减少显存占用。

脚本主要包含以下部分：
1. 环境配置与依赖导入。
2. 从 ModelScope 下载并加载量化后的模型及处理器。
3. 定义一个通用的图文推理函数，支持本地图片输入。
4. 在主程序中调用该函数，执行图像描述和视觉问答两个典型任务。
"""

# -- 1. 环境与依赖配置 --
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image, UnidentifiedImageError # 引入PIL库用于加载本地图像，并捕获可能的图像错误
from modelscope import snapshot_download

# -- 2. 下载并加载模型与处理器 --

# 从ModelScope社区下载模型文件到本地
print("正在从 ModelScope 下载模型...")
model_dir = snapshot_download('unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit')
print(f"模型已下载至: {model_dir}")

# 加载模型
# Qwen2_5_VLForConditionalGeneration 是Qwen-VL系列的模型类
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,             # 使用bfloat16以优化性能和显存，这是现代GPU的推荐做法
    device_map="auto",                      # 自动将模型分片加载到可用设备上（如GPU），实现无缝多卡支持
    attn_implementation="flash_attention_2",# [性能优化] 使用 Flash Attention 2 加速计算，节省显存，强烈推荐
    trust_remote_code=True                  # 必须设置为True，因为模型定义在远程代码中
)

# 加载处理器，用于将文本和图像转换为模型可接受的输入格式
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

# --- [可选] 高级图像处理配置 ---
# 模型默认处理的视觉token数范围是4-16384。
# 您可以根据需求设置min_pixels和max_pixels来调整图像分辨率的处理范围，
# 例如，限制在一个较小的范围内 (如256-1280个token)，以在性能和成本之间取得平衡。
# 这对于处理大量变尺寸图片或需要固定计算量的场景很有用。
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)


# -- 3. 定义推理函数 --

def get_vlm_response(image_path: str, user_prompt: str) -> str:
    """
    接收本地图片路径和用户问题，调用Qwen-VL模型生成并返回文本响应。

    Args:
        image_path (str): 本地图像文件的路径。
        user_prompt (str): 用户的文本提问。

    Returns:
        str: 模型生成的文本回答。如果图片无法加载，则返回错误信息。
    """
    # a. 加载并校验本地图像
    try:
        # 使用Pillow库打开本地图像文件
        image = Image.open(image_path)
    except FileNotFoundError:
        return f"[错误] 图片文件未找到: {image_path}"
    except UnidentifiedImageError:
        return f"[错误] 无法识别或文件已损坏: {image_path}"

    # b. 构造符合聊天格式的输入消息列表
    # 这里的 'image'可以直接传入Pillow的Image对象
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]

    # c. 将消息列表应用聊天模板，生成完整的prompt文本
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # d. 使用processor统一处理文本和图像，转换为模型输入格式
    # processor会自动处理Pillow图像对象
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt"
    )
    # 将处理好的输入数据移动到模型所在的设备（CPU/GPU）
    inputs = inputs.to(model.device)

    # e. 模型推理
    # 使用.generate()方法进行推理
    # - max_new_tokens: 控制生成文本的最大长度
    # - do_sample=False: 使用贪心解码，每次都选择概率最高的token，生成确定性的输出，适合测试和评估
    generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    # f. 结果后处理与解码
    # 从生成结果中去除输入的token部分，只保留新生成的内容
    # `generated_ids` 包含 "输入" + "回答"，我们需要切片掉 "输入" 部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 将生成的token ID解码为人类可读的文本
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0] # batch_decode返回一个列表，我们取第一个元素
    
    return response

# -- 4. 主程序执行 --
if __name__ == "__main__":
    # 定义测试用的图片路径
    # test_image = "data/file_explorer.png" 
    # test_image = "data/desktop_clean.png"
    test_image = "data/login_page.png"
    
    print("\n" + "="*50)
    print("      开始执行Qwen-VL多模态任务测试")
    print("="*50 + "\n")

    # 任务1: 图像描述 (Captioning)
    print("--- 任务1: 图像描述 (Image Captioning) ---")
    caption_prompt = "简单描述这张截图的内容。"
    caption = get_vlm_response(test_image, caption_prompt)
    print(f"图片: {test_image}")
    print(f"问题: {caption_prompt}")
    print(f"模型回答:\n{caption}\n")

    # 任务2: 视觉问答 (VQA)
    print("--- 任务2: 视觉问答 (Visual Question Answering) ---")
    vqa_prompt = "这张截图中的用户可以通过什么方式登录？"
    # vqa_prompt = "这张截图中的桌面上有多少应用程序？它们分别是什么？"
    # vqa_prompt = "这张截图中的当前位于的路径是什么？"
    answer = get_vlm_response(test_image, vqa_prompt)
    print(f"图片: {test_image}")
    print(f"问题: {vqa_prompt}")
    print(f"模型回答:\n{answer}\n")
    
    print("="*50)
    print("      任务测试完成")
    print("="*50)