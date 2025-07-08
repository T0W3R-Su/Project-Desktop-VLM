import os
import sys
import re
from PIL import Image

# 确保可以导入你的工具函数
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.model_loader import load_model_and_processor
from utils.grounding_utils import inference, draw_click_on_image  # 我们只需要推理和坐标解析
# 注意：你可能需要把你的坐标解析逻辑也抽成一个独立的函数

def parse_box_from_json(json_str):
    """
    一个简化的解析器，从你阶段二的JSON输出中提取第一个bbox。
    你需要根据你自己的json_str格式来完善它。
    """
    try:
        # 使用正则表达式从可能不完整的JSON中提取第一个bbox
        match = re.search(r'\[\s*\{\s*"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?\}\s*\]', json_str, re.DOTALL)
        if match:
            coords = [int(c) for c in match.groups()]
            # 你的格式是 [y1, x1, y2, x2]，我们统一一下
            return {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]}
    except Exception as e:
        print(f"解析坐标失败: {e}")
    return None

def get_click_coordinates(model, processor, image_path, instruction):
    """
    封装的单步定位功能：给定图片和指令，返回点击坐标。
    这是你阶段二代码的核心提炼。
    """
    system_prompt = "You are a helpful assistant. Locate the object in the image based on the instruction and provide its bounding box in JSON format."
    prompt_template = "Instruction: \"{instruction}\". Provide the JSON for the bounding box: [{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"element\"}}]"
    prompt = prompt_template.format(instruction=instruction)

    json_response, input_height, input_width = inference(model, processor, image_path, prompt, system_prompt)
    
    box = parse_box_from_json(json_response)
    if box:
        # 计算边界框的中心点作为点击坐标
        click_x = (box['x1'] + box['x2']) / 2
        click_y = (box['y1'] + box['y2']) / 2
        return ((click_x, click_y), (input_height, input_width))
    return None

def run_calculator_task(model, processor):
    """
    主Agent循环，执行计算器任务，并对每一步进行可视化。
    """
    output_dir = "output/calculator_task" # 为本次任务创建一个专门的输出文件夹

    # 1. 定义任务分解
    task_steps = [
        {"instruction": "定位按钮 '1'", "screenshot": "data/calc_01_initial.png"},
        {"instruction": "定位按钮 '2'", "screenshot": "data/calc_02_after_1.png"},
        {"instruction": "点击按钮 '3'", "screenshot": "data/calc_03_after_12.png"},
        {"instruction": "点击加号按钮 '+'", "screenshot": "data/calc_04_after_123.png"},
        {"instruction": "点击按钮 '4'", "screenshot": "data/calc_05_after_plus.png"},
        {"instruction": "点击按钮 '5'", "screenshot": "data/calc_06_after_4.png"},
        {"instruction": "点击按钮 '6'", "screenshot": "data/calc_07_after_45.png"},
        {"instruction": "点击等号按钮 '='", "screenshot": "data/calc_08_after_456.png"},
    ]

    # 2. Agent主循环
    for i, step in enumerate(task_steps):
        step_number = i + 1
        print(f"\n--- 步骤 {step_number}/{len(task_steps)} ---")
        
        current_screenshot = step["screenshot"]
        if not os.path.exists(current_screenshot):
            print(f"[错误] 截图文件不存在: {current_screenshot}。任务中断。")
            break
        print(f"👀 观察: {current_screenshot}")

        instruction = step["instruction"]
        print(f"🤔 思考: 我的下一步指令是 '{instruction}'。正在定位...")
        normalized_coords, input_coords = get_click_coordinates(model, processor, current_screenshot, instruction)
        
        if normalized_coords:
            print(f"✅ 行动: 生成指令 CLICK(x={normalized_coords[0]:.0f}, y={normalized_coords[1]:.0f})")
            
            # --- 新增的可视化步骤 ---
            output_filename = f"step_{step_number:02d}_action_on_{os.path.basename(current_screenshot)}"
            output_path = os.path.join(output_dir, output_filename)
            
            draw_click_on_image(
                image_path=current_screenshot,
                normalized_coords=normalized_coords,
                input_width=input_coords[1],
                input_height=input_coords[0],
                output_path=output_path
            )
            # --------------------------
            
        else:
            print(f"❌ 行动失败: 无法定位 '{instruction}'。")

    print("\n--- 任务流程模拟完成 ---")

def main():
    """
    程序主入口，执行计算器自动化任务。
    """
    print("--- 启动桌面智能体，任务：使用计算器计算 123 + 456 ---")
    
    # 步骤1: 加载模型 (采用单例模式，高效)
    model, processor = load_model_and_processor()

    # 步骤2: 定义任务并执行
    run_calculator_task(model, processor)

    print("\n--- 所有任务流程已成功模拟 ---")

if __name__ == '__main__':
    main()