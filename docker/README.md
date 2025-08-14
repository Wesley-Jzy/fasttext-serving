# FastText Serving - Docker 部署指南

## 🐳 Docker镜像说明

### 两种实现

| 实现 | 镜像 | 特点 | 推荐场景 |
|------|------|------|----------|
| **Python** | `fasttext-serving-python` | 使用官方FastText v0.9.2，准确性最高 | **生产环境推荐** |
| **Rust** | `fasttext-serving-rust` | 高性能，资源占用少 | 高并发场景 |

## 🚀 快速开始

### 1. 构建镜像

```bash
# 构建所有版本
./docker/build.sh -i all -t v1.0.0

# 只构建Python版本（推荐）
./docker/build.sh -i python -t v1.0.0

# 只构建Rust版本
./docker/build.sh -i rust -t v1.0.0
```

### 2. 运行容器

#### Python版本（推荐）
```bash
docker run -d \
  --name fasttext-python \
  -p 8000:8000 \
  -v /path/to/your/model:/app/models \
  fasttext-serving-python:v1.0.0 \
  --model /app/models/your-model.bin
```

#### Rust版本
```bash
docker run -d \
  --name fasttext-rust \
  -p 8000:8000 \
  -v /path/to/your/model:/app/models \
  fasttext-serving-rust:v1.0.0 \
  --model /app/models/your-model.bin \
  --workers 8
```

### 3. 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 预测测试
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '["print(\"hello world\")"]'
```

## 🔧 配置参数

### 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型文件路径 | `/app/models/model.bin` |
| `--address` | 监听地址 | `0.0.0.0` |
| `--port` | 监听端口 | `8000` |
| `--max-text-length` | 最大文本长度(字节) | `10000000` |
| `--default-threshold` | 默认预测阈值 | `0.0` |

### Rust特有参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--workers` | 工作线程数 | `4` |
| `--max-request-size` | 最大请求大小(MB) | `500` |

## 🏭 生产环境部署

### 1. 使用Docker Compose

```bash
# 设置环境变量
export MODEL_PATH=/path/to/your/models
export MODEL_FILE=your-model.bin

# 启动Python版本
docker-compose up -d fasttext-python

# 启动Rust版本
docker-compose up -d fasttext-rust

# 启动所有服务（包括负载均衡）
docker-compose --profile production up -d
```

### 2. Kubernetes部署

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fasttext-serving
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fasttext-serving
  template:
    metadata:
      labels:
        app: fasttext-serving
    spec:
      containers:
      - name: fasttext-serving
        image: fasttext-serving-python:v1.0.0
        ports:
        - containerPort: 8000
        args:
          - "--model"
          - "/app/models/model.bin"
          - "--max-text-length"
          - "20000000"
        volumeMounts:
        - name: model-volume
          mountPath: /app/models
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
```

## 📊 性能对比

| 指标 | Python版本 | Rust版本 |
|------|------------|----------|
| **准确性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **吞吐量** | 2000-3000 samples/sec | 4000-5000 samples/sec |
| **内存使用** | ~1-2GB | ~0.5-1GB |
| **启动时间** | ~10-15s | ~3-5s |
| **镜像大小** | ~800MB | ~100MB |

## 🐛 故障排查

### 常见问题

1. **模型加载失败**
   ```bash
   # 检查模型文件路径和权限
   docker run -it --rm -v /path/to/model:/app/models fasttext-serving-python:v1.0.0 ls -la /app/models
   ```

2. **内存不足**
   ```bash
   # 增加容器内存限制
   docker run --memory=2g --name fasttext-python ...
   ```

3. **端口占用**
   ```bash
   # 使用不同端口
   docker run -p 8001:8000 --name fasttext-python ...
   ```

### 查看日志

```bash
# Docker容器日志
docker logs fasttext-python

# 实时查看日志
docker logs -f fasttext-python
```

## 🔄 更新和维护

### 更新镜像

```bash
# 重新构建
./docker/build.sh -i python -t v1.0.1

# 停止旧容器
docker stop fasttext-python

# 启动新容器
docker run -d --name fasttext-python-new ...

# 删除旧容器
docker rm fasttext-python
```

### 备份和恢复

```bash
# 导出镜像
docker save fasttext-serving-python:v1.0.0 | gzip > fasttext-python-v1.0.0.tar.gz

# 导入镜像
gunzip -c fasttext-python-v1.0.0.tar.gz | docker load
```

## 📚 API文档

详见 [../docs/API.md](../docs/API.md)
