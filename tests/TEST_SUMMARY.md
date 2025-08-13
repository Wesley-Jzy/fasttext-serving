# 测试汇总表

## 🎯 测试目的快速参考

| 脚本 | 目的 | 环境要求 | 输出 | 失败原因 |
|------|------|----------|------|----------|
| `01_environment_probe.py` | 检查数据路径、Python环境 | 数据目录访问权限 | 终端输出 | 路径不存在、权限不足 |
| `02_data_explorer.py` | 分析parquet文件结构和内容 | pandas、pyarrow | `data_analysis_result.json` | 文件损坏、内存不足 |
| `03_model_validator.py` | 验证FastText服务API功能 | 服务可访问 | `model_validation_result.json` | 服务未启动、网络问题 |
| `04_service_test.py` | 测试并发性能和负载能力 | 服务可访问 | `service_test_result.json` | 超时、服务过载 |
| `05_production_client.py` | 实际处理the-stack-v2数据 | 服务+数据+输出权限 | `output/*.parquet` | 磁盘空间、网络中断 |

## 🔧 快速命令

### 环境A（FastText服务）
```bash
# 启动服务
/usr/local/bin/fasttext-serving \
  --model /mnt/project/yifan/ckpts/FT_the-stack-v2/FT_the-stack-v2.bin \
  --address 0.0.0.0 --port 8000 --workers 4

# 验证服务
curl http://localhost:8000/health
```

### 环境B（客户端测试）
```bash
# 一键运行所有测试
bash tests/run_all_tests.sh http://服务IP:8000

# 或分步执行
python3 tests/config_service_url.py http://服务IP:8000
python3 tests/01_environment_probe.py
python3 tests/02_data_explorer.py
# ... 其他测试
```

### 生产处理
```bash
python3 tests/05_production_client.py \
  --data-dir /mnt/project/yifan/data/the-stack-v2-dedup_batched_download \
  --output-dir ./output \
  --service-url http://服务IP:8000 \
  --batch-size 100 \
  --max-concurrent 10
```

## 📊 关键参数调优

| 参数 | 小数据 | 中等数据 | 大数据 |
|------|--------|----------|--------|
| batch_size | 50 | 100 | 200 |
| max_concurrent | 5 | 10 | 20 |
| timeout | 60s | 120s | 300s |

## 🚨 常见问题

1. **服务连接失败**: 检查服务是否启动、防火墙设置
2. **内存不足**: 减少batch_size和max_concurrent
3. **处理超时**: 增加timeout参数
4. **文件权限**: 确保输出目录可写
5. **parquet损坏**: 启用skip_corrupted选项
