
import os
import sys
from PIL import Image

# 将utils目录添加到Python路径，导入其中的模块
# 在项目中组织代码的常用方法
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model_loader import load_model_and_processor
from utils.grounding_utils import inference, plot_bounding_boxes

def run_visual_grounding(image_path, user_instruction, output_filename):
    """
    执行一次完整的视觉定位任务：从指令到可视化结果。
    """
    print("--- 开始视觉定位任务 ---")
    
    # 1. 加载模型和处理器 (如果已加载，会从缓存中快速返回)
    model, processor = load_model_and_processor()

    # 2. 设计"one-shot" 的Prompt，引导模型输出JSON

    system_prompt = "You are a helpful assistant that can accurately locate objects in an image based on user instructions and provide their coordinates in a JSON format."
    
    prompt_template = """
User instruction: "{instruction}"
Please provide a JSON list containing the bounding box for the requested element. The format should be:
[
  {{"bbox_2d": [x1, y1, x2, y2], "label": "your_label"}}
]
The coordinates must be normalized between 0 and 1000.
"""
    
    prompt = prompt_template.format(instruction=user_instruction)

    # 3. 调用推理函数
    #    它会返回模型的文本输出，以及模型处理时内部使用的图像尺寸
    json_response, input_height, input_width = inference(
        model, 
        processor, 
        image_path=image_path, 
        prompt=prompt,
        system_prompt=system_prompt
    )

    # 4. 可视化结果
    #    确保输出目录存在
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # 加载原始图像用于绘制
    image = Image.open(image_path)
    
    plot_bounding_boxes(
        im=image,
        json_str=json_response,
        input_width=input_width,
        input_height=input_height,
        output_path=output_path
    )
    
    print("--- 任务完成 ---")

if __name__ == '__main__':
    # --- 测试案例 1: 定位登录按钮 ---
    image_file = "data/login_page.png"
    instruction_1 = "定位登录按钮"
    output_file_1 = "grounding_login_button.png"
    run_visual_grounding(image_file, instruction_1, output_file_1)

    # --- 测试案例 2: 定位用户名输入框 ---
    instruction_2 = "定位用户名输入框"
    output_file_2 = "grounding_username_field.png"
    run_visual_grounding(image_file, instruction_2, output_file_2)

    # --- 测试案例 3: 定位关闭按钮 ---
    instruction_3 = "定位关闭按钮"
    output_file_3 = "grounding_close_button.png"
    run_visual_grounding(image_file, instruction_3, output_file_3)

    # --- 测试案例 4: 定位多个元素 ---
    file_explorer_img = "data/file_explorer.png" 
    instruction_4 = "分别定位面板中名为 Linux, Program, Github, Codefield 的文件夹。"
    output_file_4 = "grounding_all_folders.png"
    if os.path.exists(file_explorer_img):
        run_visual_grounding(file_explorer_img, instruction_4, output_file_4)

