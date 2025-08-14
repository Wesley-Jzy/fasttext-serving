# FastText Serving - Python Implementation
# 使用官方FastText v0.9.2，确保与算法一致

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（FastText编译需要）
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制Python实现
COPY implementations/python/requirements.txt .
COPY implementations/python/fasttext_server.py .

# 安装Python依赖（包括官方FastText v0.9.2）
RUN pip install --no-cache-dir -r requirements.txt

# 验证FastText安装
RUN python -c "import fasttext; print(f'FastText version: {fasttext.__version__}')"

# 创建模型目录
RUN mkdir -p /app/models

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令 - 完整参数化支持
ENTRYPOINT ["python", "fasttext_server.py"]

# 默认参数（所有参数可在启动时覆盖）
CMD ["--model", "/app/models/model.bin", \
     "--address", "0.0.0.0", \
     "--port", "8000", \
     "--max-text-length", "10000000", \
     "--default-threshold", "0.0", \
     "--default-vector-dim", "100"]

# 元数据
LABEL maintainer="fasttext-serving" \
      version="1.0.0" \
      implementation="python" \
      fasttext_version="0.9.2" \
      description="FastText Serving - Official FastText v0.9.2 implementation for maximum accuracy"
