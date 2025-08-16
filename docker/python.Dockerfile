# FastText Serving - Python Implementation
# 使用Ubuntu基础镜像 + 本地官方FastText源码

FROM ubuntu:22.04

WORKDIR /app

# 安装Python和编译依赖
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制FastText源码并安装
COPY fastText/ ./fastText/
RUN cd fastText && pip install .

# 安装基础依赖
RUN pip install flask "numpy>=1.21.0,<2.0.0" gunicorn psutil

# 复制并安装完整依赖
COPY implementations/python/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# 复制服务文件
COPY implementations/python/fasttext_server.py .
COPY start_fasttext_gunicorn.py .
COPY start_multicore_service.sh .

# 设置权限
RUN chmod +x start_multicore_service.sh

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV FASTTEXT_MODEL_PATH=/app/model/model.bin
ENV FASTTEXT_MAX_TEXT_LENGTH=10000000
ENV FASTTEXT_DEFAULT_THRESHOLD=0.0

# 暴露端口
EXPOSE 8000

# 默认启动多核服务
CMD ["./start_multicore_service.sh", "--model", "/app/model/model.bin", "--workers", "4"]