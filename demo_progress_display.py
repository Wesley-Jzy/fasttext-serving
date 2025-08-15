#!/usr/bin/env python3
"""
演示新的进度显示方式
"""

def demo_progress_display():
    """演示进度显示"""
    print("🎯 新的进度显示方式演示")
    print("=" * 60)
    
    # 模拟数据
    scenarios = [
        {
            "name": "🚀 初始阶段",
            "processed_files": 5,
            "total_files": 100,
            "processed_content_gb": 0.85,
            "total_file_gb": 45.6,
            "samples_per_sec": 1250.3,
            "gb_per_sec": 0.042
        },
        {
            "name": "📈 中期进展", 
            "processed_files": 35,
            "total_files": 100,
            "processed_content_gb": 12.4,
            "total_file_gb": 45.6,
            "samples_per_sec": 1180.7,
            "gb_per_sec": 0.038
        },
        {
            "name": "🏁 接近完成",
            "processed_files": 90,
            "total_files": 100,
            "processed_content_gb": 28.9,
            "total_file_gb": 45.6,
            "samples_per_sec": 1320.1,
            "gb_per_sec": 0.045
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        
        # 文件进度
        file_progress_pct = (scenario['processed_files'] / scenario['total_files']) * 100
        file_progress_str = f"{scenario['processed_files']}/{scenario['total_files']} ({file_progress_pct:.1f}%)"
        
        # 数据进度（不做百分比比较）
        data_progress_str = f"{scenario['processed_content_gb']:.2f}GB内容 / {scenario['total_file_gb']:.2f}GB文件"
        
        print(f"📊 进度: 文件 {file_progress_str} | "
              f"数据 {data_progress_str} | "
              f"{scenario['samples_per_sec']:.1f} samples/sec | "
              f"{scenario['gb_per_sec']:.3f} GB/s | "
              f"用时: 1800s")
    
    print(f"\n" + "=" * 60)
    print("💡 说明:")
    print("  • 文件进度: 显示百分比，表示整体完成度")
    print("  • 数据量: content(实际处理) vs 文件大小(总量估算)")
    print("  • GB/s: 基于实际content字节数，准确反映处理能力")
    print("  • 简单快速: 总量用文件大小，进度用实际content")

if __name__ == "__main__":
    demo_progress_display()
