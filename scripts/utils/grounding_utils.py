"""
Qwen-2.5-VL 官方 Cookbook spatial_understanding.ipynb 重写而来

本脚本提供了一系列与VLLM交互并可视化其输出的辅助函数。
主要功能包括：
1. `inference`: 调用模型进行推理，获取模型对图像和文本提示的响应。
2. `plot_bounding_boxes`: 解析模型输出的JSON格式边界框，并在图像上绘制出来。
3. `plot_points`: 解析模型输出的XML格式坐标点，并在图像上标记出来。
4. 辅助函数: 用于解析和清理模型原始输出的特定格式（JSON, XML）。
"""

import json
import ast
import io
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont, ImageColor

# --- 全局常量 ---

# 定义一个丰富的颜色列表，用于在图像上绘制不同的对象。
# 首先包含一组常用颜色，然后从PIL的ImageColor模块中添加更多颜色，以确保多样性。
_COLORS = [
    'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple', 'brown', 'gray',
    'beige', 'turquoise', 'cyan', 'magenta', 'lime', 'navy', 'maroon', 'teal',
    'olive', 'coral', 'lavender', 'violet', 'gold', 'silver'
] + list(ImageColor.colormap.keys())

# --- 解析函数 ---

def parse_json_from_string(text: str) -> str:
    """
    从可能包含Markdown代码块的字符串中提取纯净的JSON内容。
    
    模型有时会用 "```json\n{...}\n```" 这样的格式包裹其JSON输出，
    此函数旨在移除这些包裹，以便后续解析。

    Args:
        text (str): 包含JSON的模型原始输出字符串。

    Returns:
        str: 清理后的纯JSON字符串。
    """
    # 按行分割字符串
    lines = text.splitlines()
    for i, line in enumerate(lines):
        # 查找json代码块的起始标记
        if line.strip() == "```json":
            # 从起始标记的下一行开始，拼接所有内容
            json_content = "\n".join(lines[i+1:])
            # 找到并去除结尾的 "```"
            json_content = json_content.split("```")[0]
            return json_content.strip()
            
    # 如果没有找到 "```json" 标记，则假定整个文本就是JSON内容
    return text

def decode_xml_points(xml_text: str) -> dict | None:
    """
    解析XML格式的文本，提取其中的坐标点、替代文本(alt)和描述短语。

    Args:
        xml_text (str): 待解析的XML字符串。

    Returns:
        dict | None: 包含'points', 'alt', 'phrase'的字典，如果解析失败则返回None。
    """
    try:
        # 从字符串解析XML
        root = ET.fromstring(xml_text)
        # 根据属性数量推断点的个数（假设属性总是成对的x,y，外加一个alt属性）
        num_points = (len(root.attrib) - 1) // 2
        points = []
        for i in range(num_points):
            # 动态获取 x{i+1} 和 y{i+1} 属性
            x = root.attrib.get(f'x{i+1}')
            y = root.attrib.get(f'y{i+1}')
            if x is not None and y is not None:
                points.append([x, y])
        
        # 获取 'alt' 属性和标签内的文本
        alt = root.attrib.get('alt')
        phrase = root.text.strip() if root.text else None
        
        return {
            "points": points,
            "alt": alt,
            "phrase": phrase
        }
    except Exception as e:
        # 如果解析过程中出现任何错误，打印错误信息并返回None
        print(f"[-] XML解析失败: {e}")
        return None

# --- 可视化函数 ---

def plot_bounding_boxes(im: Image.Image, json_str: str, input_width: int, input_height: int, output_path: str = None):
    """
    在图像上绘制边界框和标签。
    该函数会解析JSON字符串，将归一化的坐标转换为绝对坐标，并用不同颜色绘制。

    Args:
        im (Image.Image): Pillow图像对象。
        json_str (str): 包含边界框信息的JSON格式字符串。
        input_width (int): 模型处理图像时所见的宽度（用于坐标归一化）。
        input_height (int): 模型处理图像时所见的高度（用于坐标归一化）。
        output_path (str, optional): 如果提供，则将绘制后的图像保存到此路径。否则，直接显示图像。
    """
    original_width, original_height = im.size
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()

    # 步骤1: 清理并解析JSON字符串
    clean_json_str = parse_json_from_string(json_str)
    
    try:
        # 使用ast.literal_eval安全地评估字符串为Python对象（列表、字典等）
        # 它比json.loads更容忍一些轻微的语法错误，但仍比eval安全得多。
        bounding_boxes = ast.literal_eval(clean_json_str)
    except Exception as e:
        print(f"[-] 使用 ast.literal_eval 解析JSON失败。错误: {e}")
        # 容错处理：大语言模型有时会因输出截断导致JSON不完整。
        # 这里尝试一种常见的修复方法：找到最后一个有效的 '}' 并补全 ']'
        try:
            # 找到最后一个 `"}` 出现的位置，并截取到其后
            end_idx = clean_json_str.rfind('"}') + len('"}')
            # 假设原始结构是 `[{"key": "value"}, ...]`，尝试在末尾补上 `]`
            truncated_text = clean_json_str[:end_idx] + "]"
            bounding_boxes = ast.literal_eval(truncated_text)
            print("[+] 容错解析成功！")
        except Exception as e2:
            print(f"[-] 容错解析同样失败。错误: {e2}")
            # 如果所有解析都失败，则放弃绘制，直接保存或显示原图以便调试。
            if output_path:
                im.save(output_path)
                print(f"[!] 因解析失败，已将原始图像保存至 {output_path}")
            else:
                im.show()
            return # 提前退出函数

    # 步骤2: 遍历每个边界框并绘制
    for i, box_data in enumerate(bounding_boxes):
        # 从颜色列表中循环选择颜色
        color = _COLORS[i % len(_COLORS)]

        # 步骤3: 坐标转换
        # 模型输出的bbox_2d是归一化坐标 [x1, y1, x2, y2]，范围在[0, input_width/input_height]
        # 需要将其转换为原始图像上的绝对像素坐标。
        y1_norm, x1_norm, y2_norm, x2_norm = box_data["bbox_2d"][1], box_data["bbox_2d"][0], box_data["bbox_2d"][3], box_data["bbox_2d"][2]

        abs_y1 = int(y1_norm / input_height * original_height)
        abs_x1 = int(x1_norm / input_width * original_width)
        abs_y2 = int(y2_norm / input_height * original_height)
        abs_x2 = int(x2_norm / input_width * original_width)
        
        # 确保(x1, y1)是左上角, (x2, y2)是右下角
        if abs_x1 > abs_x2: abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2: abs_y1, abs_y2 = abs_y2, abs_y1

        # 步骤4: 绘制矩形框
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), 
            outline=color, 
            width=4
        )

        # 步骤5: 绘制标签文本
        if "label" in box_data:
            label = box_data["label"]
            # 在框的左上角内侧绘制标签文本
            draw.text((abs_x1 + 8, abs_y1 + 6), label, fill=color, font=font)

    # 步骤6: 保存或显示结果
    if output_path:
        im.save(output_path)
        print(f"[+] 带有边界框的图像已保存至: {output_path}")
    else:
        im.show()

def plot_points(im: Image.Image, xml_str: str, input_width: int, input_height: int):
    """
    在图像上绘制坐标点和描述。
    该函数会解析XML字符串，将归一化的坐标转换为绝对坐标，并用不同颜色标记。

    Args:
        im (Image.Image): Pillow图像对象。
        xml_str (str): 包含坐标点信息的XML格式字符串。
        input_width (int): 模型处理图像时所见的宽度。
        input_height (int): 模型处理图像时所见的高度。
    """
    original_width, original_height = im.size
    draw = ImageDraw.Draw(im)
    font = ImageFont.load_default()
    
    # 步骤1: 清理XML字符串，移除Markdown代码块标记
    clean_xml_str = xml_str.replace('```xml', '').replace('```', '').strip()
    
    # 步骤2: 解析XML数据
    data = decode_xml_points(clean_xml_str)
    if data is None:
        # 如果解析失败，直接显示原图
        print("[-] XML数据解析失败，无法绘制点。")
        im.show()
        return

    points = data.get('points', [])
    description = data.get('phrase', '')

    # 步骤3: 遍历每个点并绘制
    for i, point in enumerate(points):
        # 循环选择颜色
        color = _COLORS[i % len(_COLORS)]
        
        # 步骤4: 坐标转换
        # point[0]是x, point[1]是y，均为字符串形式的归一化坐标
        x_norm, y_norm = int(point[0]), int(point[1])
        abs_x = int(x_norm / input_width * original_width)
        abs_y = int(y_norm / input_height * original_height)
        
        # 步骤5: 绘制点（一个小的实心圆）和描述文本
        radius = 5  # 增大半径使其更明显
        draw.ellipse(
            [(abs_x - radius, abs_y - radius), (abs_x + radius, abs_y + radius)], 
            fill=color
        )
        if description:
            draw.text((abs_x + 8, abs_y - 6), f"({i+1}) {description}", fill=color, font=font)

    # 步骤6: 显示结果
    im.show()


# --- 模型推理函数 ---

def inference(
    model, 
    processor, 
    image_path: str, 
    prompt: str, 
    system_prompt: str = "You are a helpful assistant.", 
    max_new_tokens: int = 1024
) -> tuple[str, int, int]:
    """
    使用指定的VLLM模型和处理器执行端到端的推理。

    Args:
        model: 已加载的VLLM模型。
        processor: 对应的处理器，用于文本和图像的预处理。
        image_path (str): 本地图像文件的路径。
        prompt (str): 向模型提出的文本问题或指令。
        system_prompt (str, optional): 系统提示，用于设定模型的角色或行为。
        max_new_tokens (int, optional): 模型生成新文本的最大长度。

    Returns:
        tuple[str, int, int]:
            - str: 模型生成的文本输出。
            - int: 模型内部处理时使用的图像高度。
            - int: 模型内部处理时使用的图像宽度。
    """
    # 1. 加载图像
    image = Image.open(image_path)

    # 2. 构建符合模型聊天模板的输入消息格式
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image}
            ]
        }
    ]

    # 3. 应用聊天模板，生成模型可以直接处理的文本输入
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("--- 模型输入文本 ---\n", prompt_text)

    # 4. 使用处理器对文本和图像进行预处理，转换为模型所需的张量格式
    inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt").to(model.device)

    # 5. 执行模型生成（推理）
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    
    # 6. 从输出中分离出新生成的部分
    generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    
    # 7. 将生成的token IDs解码为可读的文本
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("\n--- 模型原始输出 ---\n", output_text[0])

    # 8. 获取模型处理图像时内部使用的网格尺寸，并计算出归一化坐标系的基准宽高。
    # `image_grid_thw` 包含了图像被切分成网格的信息 [T, H, W]。
    # 这里的 `14` 很可能是模型使用的patch_size(图像块大小)，这是一个与模型架构相关的硬编码值。
    # 不同的模型可能有不同的patch_size。
    input_height = inputs['image_grid_thw'][0][1] * 14
    input_width = inputs['image_grid_thw'][0][2] * 14

    return output_text[0], input_height, input_width