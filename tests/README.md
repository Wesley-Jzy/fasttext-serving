# FastText Serving 马来环境测试

这个目录包含在马来集群环境中测试和开发的脚本。

## 🔧 开发工作流

1. **本地开发**: 在这个环境修改代码
2. **Git同步**: `git add . && git commit -m "update" && git push`
3. **马来执行**: 在马来环境 `git pull && python3 tests/xxx.py`
4. **结果反馈**: 复制执行结果回到这边继续开发

## 📁 文件结构

```
tests/
├── README.md                 # 本文件
├── 01_environment_probe.py   # 环境探测脚本
├── 02_data_explorer.py       # 数据结构探索
├── 03_model_validator.py     # 模型加载验证
├── 04_service_test.py        # 服务连通性测试
├── 05_production_client.py   # 生产级数据清洗客户端
└── requirements.txt          # 测试环境依赖
```

## 🎯 执行顺序

按数字顺序执行脚本，每个脚本会输出关键信息用于下一步开发。
