#!/usr/bin/env python3
"""
FastText Gunicorn启动器
支持多进程，充分利用多核CPU
"""

import os
import sys
import argparse
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent / "implementations" / "python"))

from fasttext_server import FastTextServer, create_app

# 全局变量，用于Gunicorn
server = None
app = None

def create_application():
    """创建WSGI应用，供Gunicorn使用"""
    global server, app
    
    # 从环境变量获取配置
    model_path = os.environ.get('FASTTEXT_MODEL_PATH', 
                               '/mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin')
    max_text_length = int(os.environ.get('FASTTEXT_MAX_TEXT_LENGTH', '10000000'))
    default_threshold = float(os.environ.get('FASTTEXT_DEFAULT_THRESHOLD', '0.0'))
    default_vector_dim = int(os.environ.get('FASTTEXT_DEFAULT_VECTOR_DIM', '100'))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading FastText model: {model_path}")
    
    # 创建服务器实例
    server = FastTextServer(
        model_path=model_path,
        max_text_length=max_text_length,
        default_threshold=default_threshold,
        default_vector_dim=default_vector_dim
    )
    
    # 创建Flask应用
    app = create_app(server)
    
    print(f"FastText service initialized (PID: {os.getpid()})")
    
    return app

# 为Gunicorn提供application对象
application = create_application()

if __name__ == "__main__":
    # 直接运行时的参数解析
    parser = argparse.ArgumentParser(description='FastText Gunicorn启动器')
    parser.add_argument('--model', required=True, help='FastText模型文件路径')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    parser.add_argument('--workers', type=int, default=8, help='Worker进程数')
    parser.add_argument('--timeout', type=int, default=300, help='请求超时时间(秒)')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['FASTTEXT_MODEL_PATH'] = args.model
    
    # 构建Gunicorn命令
    cmd = f"""gunicorn -w {args.workers} -b 0.0.0.0:{args.port} \\
        --timeout {args.timeout} \\
        --worker-class sync \\
        --max-requests 1000 \\
        --max-requests-jitter 100 \\
        --preload \\
        start_fasttext_gunicorn:application"""
    
    print(f"启动命令: {cmd}")
    print(f"Worker进程数: {args.workers}")
    print(f"服务端口: {args.port}")
    print(f"模型路径: {args.model}")
    
    # 执行命令
    os.system(cmd)
