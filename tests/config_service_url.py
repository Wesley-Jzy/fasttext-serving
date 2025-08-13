#!/usr/bin/env python3
"""
快速配置测试脚本中的服务URL
"""
import sys
import re

def update_service_url(new_url: str):
    """更新测试脚本中的服务URL"""
    files_to_update = [
        ("tests/03_model_validator.py", r'service_url = "[^"]*"  # TODO: 修改为实际服务地址[^"]*'),
        ("tests/04_service_test.py", r'base_url = "[^"]*"  # TODO: 修改为实际服务地址'),
    ]
    
    print(f"🔧 更新服务URL为: {new_url}")
    
    for file_path, pattern in files_to_update:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if '03_model_validator.py' in file_path:
                new_line = f'service_url = "{new_url}"  # 已配置的服务地址'
            else:
                new_line = f'base_url = "{new_url}"  # 已配置的服务地址'
            
            updated_content = re.sub(pattern, new_line, content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"✅ 已更新: {file_path}")
            
        except Exception as e:
            print(f"❌ 更新失败 {file_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("用法: python3 tests/config_service_url.py <服务URL>")
        print("例如: python3 tests/config_service_url.py http://10.1.1.100:8000")
        sys.exit(1)
    
    service_url = sys.argv[1]
    update_service_url(service_url)
    print("\n🎯 配置完成！现在可以运行测试脚本了")

if __name__ == "__main__":
    main()
