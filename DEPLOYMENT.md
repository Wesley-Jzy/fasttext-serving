# FastText Serving 部署指南

## 📦 Docker镜像信息

**镜像地址**: `hub.anuttacon.com/infra/fasttext-serving:latest`  
**镜像大小**: 106MB  
**构建状态**: ✅ 成功构建并推送  

## 🚀 平台部署参数

### K8s 部署命令参数

```bash
# 容器启动参数
--model /app/models/cooking.model.bin \
--address 0.0.0.0 \
--port 8000 \
--workers 4 \
--max-request-size 500
```

### 环境变量（可选）
```bash
RUST_LOG=fasttext_serving=info
```

### 资源要求
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi" 
    cpu: "2000m"
```

### 健康检查配置
```yaml
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
```

## 🧪 部署后测试

获得服务URL后，使用以下方法进行测试：

### 方法1: 快速测试脚本（推荐）

```bash
cd examples
python test_service.py http://your-k8s-service-url:8000
```

这个脚本会自动执行：
- ✅ 健康检查
- ✅ 基础预测功能测试  
- ✅ 大批量处理测试（1000个样本）
- ✅ 性能统计和结果分析

### 方法2: 手动测试

```bash
# 1. 健康检查
curl http://your-k8s-service-url:8000/health

# 2. 基础预测测试
curl -X POST -H 'Content-Type: application/json' \
  --data '["Which baking dish is best to bake a banana bread?", "Why not put knives in the dishwasher?"]' \
  http://your-k8s-service-url:8000/predict

# 3. 客户端测试
cd examples  
python client.py --url http://your-k8s-service-url:8000 --use-test-data
```

### 方法3: 高级测试

```bash
cd examples

# 性能测试
time python client.py \
  --url http://your-k8s-service-url:8000 \
  --use-test-data \
  --batch-size 2000 \
  --max-concurrent 4

# 并发压力测试
for i in {1..5}; do
  python client.py \
    --url http://your-k8s-service-url:8000 \
    --use-test-data \
    --batch-size 500 \
    --max-concurrent 2 \
    --output "stress_test_$i.json" &
done
wait
```

## 📊 预期测试结果

### 健康检查
```json
{"status":"healthy","model_loaded":true}
```

### 预测结果示例
```json
[
  [["baking"], [0.7152988]], 
  [["equipment"], [0.73479545]]
]
```

### 性能指标
- **单次预测延迟**: < 100ms
- **批量处理吞吐量**: 500-2000 samples/sec
- **内存使用**: < 2GB
- **错误率**: < 5%

## 🛠️ 服务配置说明

### 支持的参数
- `--model`: 模型文件路径（必需）
- `--address`: 监听地址（默认: 127.0.0.1）
- `--port`: 监听端口（默认: 8000）  
- `--workers`: 工作线程数（默认: CPU核数）
- `--max-request-size`: 最大请求大小MB（默认: 500）
- `--grpc`: 启用gRPC而非HTTP（可选）

### API 端点
- `GET /health`: 健康检查
- `POST /predict`: 文本分类预测
- `POST /sentence-vector`: 句子向量化（可选）

### Cooking模型分类
当前模型可识别的类别：
- **baking**: 烘焙相关问题
- **equipment**: 厨具设备问题  
- **food-safety**: 食品安全问题
- **preparation**: 食材准备问题
- **cooking-method**: 烹饪方法问题

## 🔧 故障排除

### 常见问题

#### 1. 服务无法启动
- 检查模型文件是否存在和可读
- 检查端口是否被占用
- 查看容器日志: `kubectl logs <pod-name>`

#### 2. 健康检查失败
- 确认服务已完全启动（可能需要30秒）
- 检查网络连接和防火墙
- 验证端口映射配置

#### 3. 预测请求超时
- 检查请求大小是否超过限制
- 增加客户端超时时间
- 考虑减少批处理大小

#### 4. 内存不足
- 减少worker数量
- 降低max-request-size参数
- 增加K8s资源限制

### 日志等级
```bash
# 详细日志
RUST_LOG=fasttext_serving=debug

# 正常日志  
RUST_LOG=fasttext_serving=info

# 错误日志
RUST_LOG=fasttext_serving=error
```

## ✅ 验收标准

部署成功的判断标准：

1. **健康检查**: 返回200状态码和正确JSON
2. **功能测试**: 能正确分类cooking相关问题
3. **性能测试**: 吞吐量 > 500 samples/sec
4. **稳定性**: 连续运行1小时无崩溃
5. **错误率**: < 5%的预测错误

## 🎯 下一步

部署成功后，你可以：

1. **替换模型**: 将cooking模型替换为你的代码质量分类模型
2. **扩展客户端**: 修改`examples/client.py`适配你的数据格式
3. **性能调优**: 根据实际负载调整参数
4. **监控告警**: 集成到你的监控系统
5. **水平扩展**: 增加Pod副本数量处理更大规模数据

---

**部署完成时间**: $(date)  
**镜像版本**: hub.anuttacon.com/infra/fasttext-serving:latest  
**测试状态**: ✅ 本地验证通过 