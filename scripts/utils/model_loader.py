"""
本模块负责Qwen2.5-VL模型的下载与加载。

功能：
1. 从 ModelScope 下载指定版本的 Qwen-VL 模型文件。
2. 使用 Transformers 加载量化后的模型和对应的处理器。
3. 实现单例模式（Singleton-like），确保在整个应用生命周期中模型只被加载一次，
   避免重复占用显存和加载时间。
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
import os

# 在模块级别定义变量，用于缓存已加载的模型和处理器
_model = None
_processor = None

# 指定模型ID
MODEL_ID = 'unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit' # 原脚本使用的bnb-4bit版本

def load_model_and_processor():
    """
    加载并返回Qwen-VL模型和处理器。

    如果模型和处理器已经加载，则直接从缓存中返回，否则执行下载和加载过程。

    Returns:
        tuple: (model, processor)
               - model: 加载好的 Qwen2_5_VLForConditionalGeneration 模型实例。
               - processor: 加载好的 AutoProcessor 处理器实例。
    """
    global _model, _processor

    # 检查是否已经加载过，如果加载过则直接返回
    if _model is not None and _processor is not None:
        print("模型和处理器已加载，直接从缓存返回。")
        return _model, _processor

    # --- 下载模型 ---
    print(f"正在从 ModelScope 下载模型: {MODEL_ID}...")
    # 使用 atexit 确保在脚本退出时能看到下载进度条的完整输出
    model_dir = snapshot_download(MODEL_ID)
    print(f"模型已下载至: {model_dir}")

    # --- 加载模型 ---
    print("正在加载模型到设备...")
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    print("模型加载完成。")

    # --- 加载处理器 ---
    print("正在加载处理器...")
    _processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    print("处理器加载完成。")
    
    return _model, _processor

if __name__ == '__main__':
    # 这个部分用于直接运行此文件时进行测试，确保加载功能正常
    print("正在测试模型加载功能...")
    model, processor = load_model_and_processor()
    
    # 打印模型和处理器的信息以验证
    print("\n--- 验证加载结果 ---")
    print(f"模型类型: {type(model)}")
    print(f"处理器类型: {type(processor)}")
    print(f"模型所在设备: {model.device}")
    
    # 尝试再次调用，验证缓存机制
    print("\n--- 测试缓存机制 ---")
    load_model_and_processor()
    print("\n加载测试完成。")