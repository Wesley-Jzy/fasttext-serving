# 测试执行指南

## 🏗️ 环境设置

### 环境A：FastText服务环境（马来集群）
- **镜像**: FastText Serving Docker镜像
- **用途**: 运行FastText推理服务
- **需要**: 模型文件挂载、端口暴露

### 环境B：客户端测试环境（马来集群）
- **镜像**: PyTorch镜像（或纯Python环境）
- **用途**: 运行数据处理和API测试
- **需要**: 数据目录挂载、网络访问

## 📋 测试执行流程

### 第1步：启动FastText服务（环境A）
```bash
# 在FastText镜像环境中运行
/usr/local/bin/fasttext-serving \
  --model /mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin \
  --address 0.0.0.0 \
  --port 8000 \
  --workers 4
```
**目的**: 启动FastText推理服务，提供HTTP API
**验证**: 访问 `http://服务地址:8000/health` 返回正常

### 第2步：客户端环境准备（环境B）
```bash
# 在PyTorch镜像环境中执行
git clone <你的repo地址>
cd fasttext-serving
pip install -r tests/requirements.txt
```

### 第3步：环境探测测试
```bash
python3 tests/01_environment_probe.py
```
**目的**: 检查数据路径、Python环境、网络连接
**预期**: 确认数据目录存在、依赖安装正常
**如果失败**: 安装缺失依赖，检查数据路径权限

### 第4步：数据结构分析
```bash
python3 tests/02_data_explorer.py
```
**目的**: 分析parquet文件结构、content列特征、样本分布
**预期**: 生成 `data_analysis_result.json`，了解数据规模
**如果失败**: 检查parquet文件权限，确认pyarrow安装

### 第5步：模型API验证
```bash
# 需要修改脚本中的service_url为实际地址
python3 tests/03_model_validator.py
```
**目的**: 通过HTTP API测试模型二分类功能
**预期**: 确认标签格式、性能表现
**如果失败**: 检查服务连接，确认API格式

### 第6步：服务性能测试
```bash
python3 tests/04_service_test.py
```
**目的**: 测试并发性能、大批量处理能力
**预期**: 获得吞吐量数据，验证负载均衡
**如果失败**: 调整并发参数、超时设置

### 第7步：生产数据处理
```bash
python3 tests/05_production_client.py \
  --data-dir /mnt/project/yifan/data/the-stack-v2-dedup_batched_download \
  --output-dir ./output \
  --service-url http://实际服务地址:8000 \
  --batch-size 100 \
  --max-concurrent 10
```
**目的**: 实际处理the-stack-v2数据，验证端到端流程
**预期**: 生成处理后的parquet文件
**如果失败**: 根据错误调整参数

## 🔧 可能需要的调整

### 1. 服务URL配置
在以下文件中修改服务地址：
- `tests/03_model_validator.py` 第152行
- `tests/04_service_test.py` 第218行

### 2. 数据路径检查
如果数据路径不同，修改：
- `tests/01_environment_probe.py` 第42行
- `tests/02_data_explorer.py` 第243行
- `tests/03_model_validator.py` 第149行

### 3. 性能参数调整
根据实际环境调整：
- 批次大小（batch_size）
- 并发数（max_concurrent）
- 超时时间（timeout）

## 📊 输出文件说明

| 文件 | 来源 | 内容 |
|------|------|------|
| `data_analysis_result.json` | 02_data_explorer.py | 数据结构分析 |
| `model_validation_result.json` | 03_model_validator.py | 模型功能验证 |
| `service_test_result.json` | 04_service_test.py | 服务性能测试 |
| `output/*.parquet` | 05_production_client.py | 处理后的数据 |
| `processing_report.json` | 05_production_client.py | 处理统计报告 |
