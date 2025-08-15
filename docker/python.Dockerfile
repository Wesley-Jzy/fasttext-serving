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

# 安装其他依赖
RUN pip install flask "numpy>=1.21.0,<2.0.0"

# 复制服务文件
COPY implementations/python/fasttext_server.py .

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 默认bash启动
CMD ["/bin/bash"]