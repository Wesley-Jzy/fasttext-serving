# FastText Serving - 大规模代码质量分类平台

[![GitHub Actions](https://github.com/messense/fasttext-serving/workflows/CI/badge.svg)](https://github.com/messense/fasttext-serving/actions?query=workflow%3ACI)
[![Crates.io](https://img.shields.io/crates/v/fasttext-serving.svg)](https://crates.io/crates/fasttext-serving)
[![Docker Pulls](https://img.shields.io/docker/pulls/messense/fasttext-serving)](https://hub.docker.com/r/messense/fasttext-serving)

专为 **The-Stack-v2** 大规模代码数据集设计的 FastText 推理服务平台，支持 **千万级代码样本** 的高性能质量分类处理。

## 🎯 核心功能

- **🚀 高性能推理**: Python/Rust双实现，2500+ samples/sec吞吐量
- **📊 大规模处理**: 支持TB级数据的增量处理和断点续传
- **🔧 开箱即用**: 一键部署，自动性能调优
- **⚡ 实时监控**: 完整的处理进度和性能统计
- **🛡️ 生产就绪**: 容错恢复、健康检查、日志完备

---

## 🚀 快速开始

### 1️⃣ 获取代码

```bash
git clone git@ai-git.anuttacon.org:wesley.jiang/fasttext-serving.git
cd fasttext-serving
```

### 2️⃣ 服务端部署

#### Docker部署（推荐）

```bash
# 构建镜像
./docker/build.sh -i python -t v1.2.0

#### 直接运行

```bash
# 安装依赖
pip3 install -r implementations/python/requirements.txt

# 启动服务（后台运行）
nohup python3 implementations/python/fasttext_server.py \
  --model /mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin \
  --address 0.0.0.0 \
  --port 8000 \
  --max-text-length 20000000 \
  --default-threshold 0.0 \
  > fasttext_server.log 2>&1 &

# 检查服务状态
curl http://localhost:8000/health
```

### 3️⃣ 客户端使用

#### 🔧 快速设置

```bash
# 运行交互式设置向导
./tools/quick_setup.sh
```

#### 📋 手动配置

```bash
# 1. 安装客户端依赖
pip3 install pandas aiohttp pyarrow

# 2. 真实数据性能测试（推荐 - 使用实际数据）
python3 tools/real_data_performance_tester.py \\
  --api-url http://your-api-server-ip:port \\
  --data-dir /path/to/the-stack-v2 \\
  --output real_perf_results.json

# 3. 或使用合成数据性能测试
python3 tools/api_performance_tester.py \\
  --api-url http://your-api-server-ip:port \\
  --output synthetic_perf_results.json

# 4. 开始处理数据
python3 client/the_stack_processor.py \\
  --data-dir /path/to/the-stack-v2 \\
  --output-dir /path/to/results \\
  --api-url http://your-api-server-ip:port \\
  --max-concurrent 80 \\
  --batch-size 200 \\
  --resume
```

---

## 📖 详细使用指南

### 🖥️ 服务端配置

#### 启动参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--model` | 模型文件路径 | 必需 | `/app/models/fasttext.bin` |
| `--address` | 监听地址 | `0.0.0.0` | `127.0.0.1` |
| `--port` | 监听端口 | `8000` | `9000` |
| `--max-text-length` | 最大文本长度(字节) | `10000000` | `5000000` |
| `--default-threshold` | 默认预测阈值 | `0.0` | `0.5` |


### 🎛️ 客户端配置

#### 性能调优

**真实数据测试（推荐）** - 使用实际数据获取最准确的性能配置：

```bash
python3 tools/real_data_performance_tester.py \\
  --api-url http://api-server:8000 \\
  --data-dir /path/to/the-stack-v2 \\
  --output real_perf_results.json

# 查看推荐配置
cat real_perf_results.json | jq '.best_configuration.overall_best_throughput'
```

**合成数据测试** - 快速获取基础性能参考：

```bash
python3 tools/api_performance_tester.py \\
  --api-url http://api-server:8000 \\
  --test-duration 60 \\
  --output perf_results.json

# 查看推荐配置
cat perf_results.json | jq '.best_configuration.best_overall'
```

#### 处理参数详解

| 参数 | 说明 | 推荐值 | 备注 |
|------|------|--------|------|
| `--max-concurrent` | 并发请求数 | `50-100` | 根据API性能测试结果 |
| `--batch-size` | 批处理大小 | `200-500` | 平衡延迟和吞吐量 |
| `--timeout` | 请求超时(秒) | `30` | 根据网络情况调整 |
| `--stability-window` | 文件稳定窗口(秒) | `30` | 检测文件写入完成 |
| `--resume` | 断点续传 | `启用` | 支持中断后继续 |
| `--save-format` | 输出格式 | `parquet` | 或选择 `json` |

#### 大规模处理示例

```bash
# 高性能配置 - 适合专用服务器
python3 client/the_stack_processor.py \\
  --data-dir /mnt/data/the-stack-v2 \\
  --output-dir /mnt/results \\
  --api-url http://api-cluster:8000 \\
  --max-concurrent 100 \\
  --batch-size 500 \\
  --timeout 60 \\
  --resume \\
  --log-level INFO

# 保守配置 - 适合共享环境
python3 client/the_stack_processor.py \\
  --data-dir /path/to/data \\
  --output-dir /path/to/results \\
  --api-url http://localhost:8000 \\
  --max-concurrent 20 \\
  --batch-size 100 \\
  --resume
```

---

## 🐳 Docker使用

### 镜像构建

```bash
./docker/build.sh -i python -t v1.2.0

```


### 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MODEL_PATH` | 模型目录路径 | `./models` |
| `MODEL_FILE` | 模型文件名 | `model.bin` |
| `MAX_TEXT_LENGTH` | 最大文本长度 | `10000000` |
| `DEFAULT_THRESHOLD` | 默认阈值 | `0.0` |
| `WORKERS` | 工作线程数(Rust) | `4` |

---

## 📊 数据格式

### 输入数据格式

The-Stack-v2 Parquet文件应包含以下列：

```json
{
  "content": "代码文本内容",
  "blob_id": "文件标识符", 
  "path": "文件路径",
  "repo_name": "仓库名称",
  "language": "编程语言",
  "size": 文件大小,
  "ext": "文件扩展名"
}
```

### 输出数据格式

处理后的结果包含原始数据 + 质量分类：

```json
{
  "blob_id": "原始文件标识符",
  "path": "原始文件路径", 
  "repo_name": "原始仓库名称",
  "language": "编程语言",
  "size": 文件大小,
  "ext": "文件扩展名",
  
  "quality_labels": ["__label__1", "__label__0"],
  "quality_scores": [0.8234, 0.1766],
  "quality_prediction": "__label__1",
  "quality_confidence": 0.8234,
  
  "content_length": 1024,
  "processed_at": "2024-01-01T12:00:00"
}
```

---

## 🔧 API接口

### 预测接口

```http
POST /predict?k=2&threshold=0.0
Content-Type: application/json

[
  "def hello_world():\\n    print('Hello!')",
  "SELECT * FROM users;",
  "console.log('Hello, JS!');"
]
```

**响应:**
```json
[
  {
    "labels": ["__label__1", "__label__0"],
    "scores": [0.8234, 0.1766]
  },
  {
    "labels": ["__label__0", "__label__1"], 
    "scores": [0.9102, 0.0898]
  }
]
```

### 健康检查

```http
GET /health
```

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "implementation": "python",
  "version": "1.2.0"
}
```


## 📝 开发者信息

### 项目结构

```
fasttext-serving/
├── implementations/
│   ├── python/           # Python实现
│   └── rust/             # Rust实现
├── client/               # 客户端工具
├── tools/                # 辅助工具
├── docker/               # Docker配置
└── docs/                 # 文档
```

### 扩展开发

```bash
# 添加新的预处理逻辑
# 编辑: client/the_stack_processor.py -> preprocess_content()

# 自定义输出格式
# 编辑: client/the_stack_processor.py -> postprocess_results()

# 添加新的性能指标
# 编辑: tools/api_performance_tester.py
```

---

