# FastText Serving - Rust Implementation
# 多阶段构建：编译 + 运行时

# ================================
# 构建阶段
# ================================
FROM rust:1.70 as builder

WORKDIR /app

# 安装protobuf编译器
RUN apt-get update && \
    apt-get install -y protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*

# 复制Rust源码
COPY implementations/rust/Cargo.toml implementations/rust/Cargo.lock ./
COPY implementations/rust/build.rs ./
COPY implementations/rust/src/ ./src/
COPY implementations/rust/proto/ ./proto/

# 编译Release版本（带标签修复）
RUN cargo build --release --features http

# ================================
# 运行时阶段
# ================================
FROM ubuntu:20.04

# 安装运行时依赖
RUN apt-get update && \
    apt-get install -y ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 从builder复制二进制文件
COPY --from=builder /app/target/release/fasttext-serving /usr/local/bin/fasttext-serving

# 创建模型目录
RUN mkdir -p /app/models

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV RUST_LOG=fasttext_serving=info

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 入口点 - 完整参数化支持
ENTRYPOINT ["/usr/local/bin/fasttext-serving"]

# 默认参数（所有参数可在启动时覆盖）
CMD ["--model", "/app/models/model.bin", \
     "--address", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--max-request-size", "500", \
     "--max-text-length", "10000000", \
     "--default-threshold", "0.0", \
     "--default-vector-dim", "100"]

# 元数据
LABEL maintainer="fasttext-serving" \
      version="1.0.0" \
      implementation="rust" \
      description="FastText Serving - High-performance Rust implementation with fixed label processing"
