#!/usr/bin/env python3
"""
增量文件检测器 - 判断parquet文件是否写入完成
"""
import os
import time
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import logging

class IncrementalFileDetector:
    """增量文件检测器"""
    
    def __init__(self, stability_window: int = 30):
        """
        Args:
            stability_window: 文件稳定性窗口(秒)，文件在此时间内无变化才认为完整
        """
        self.stability_window = stability_window
        self.file_states = {}  # 文件状态缓存
        
    def is_file_ready(self, file_path: Path) -> Tuple[bool, str]:
        """
        检测文件是否写入完成并可处理
        
        Returns:
            (is_ready, reason)
        """
        if not file_path.exists():
            return False, "file_not_exists"
            
        try:
            # 1. 检查文件大小稳定性
            current_size = file_path.stat().st_size
            current_mtime = file_path.stat().st_mtime
            
            # 空文件直接跳过
            if current_size == 0:
                return False, "empty_file"
            
            # 检查文件是否在稳定窗口内有变化
            if str(file_path) in self.file_states:
                last_size, last_mtime, last_check = self.file_states[str(file_path)]
                
                # 文件发生了变化，重新计时
                if current_size != last_size or current_mtime != last_mtime:
                    self.file_states[str(file_path)] = (current_size, current_mtime, time.time())
                    return False, "file_still_changing"
                
                # 文件在稳定窗口内无变化
                if time.time() - last_check >= self.stability_window:
                    # 尝试读取验证
                    return self._validate_parquet_readable(file_path)
                else:
                    return False, f"waiting_stability_{int(self.stability_window - (time.time() - last_check))}s"
            else:
                # 首次检测，记录状态
                self.file_states[str(file_path)] = (current_size, current_mtime, time.time())
                return False, "first_detection"
                
        except Exception as e:
            return False, f"check_error_{str(e)}"
    
    def _validate_parquet_readable(self, file_path: Path) -> Tuple[bool, str]:
        """验证parquet文件是否可读"""
        try:
            # 尝试读取文件头信息（不加载全部数据）
            df_info = pd.read_parquet(file_path, engine='pyarrow')
            
            # 基本完整性检查
            if len(df_info) == 0:
                return False, "empty_dataframe"
                
            # 检查必需列
            required_columns = ['content']  # 根据实际需求调整
            missing_cols = [col for col in required_columns if col not in df_info.columns]
            if missing_cols:
                return False, f"missing_columns_{missing_cols}"
                
            return True, "ready"
            
        except Exception as e:
            return False, f"parquet_error_{str(e)}"
    
    def scan_ready_files(self, data_dir: Path, pattern: str = "*.parquet") -> List[Path]:
        """扫描目录中已准备好的文件"""
        ready_files = []
        
        for file_path in data_dir.glob(pattern):
            is_ready, reason = self.is_file_ready(file_path)
            
            if is_ready:
                ready_files.append(file_path)
                logging.info(f"✅ Ready: {file_path.name}")
            else:
                logging.debug(f"⏳ Waiting: {file_path.name} ({reason})")
        
        return ready_files
    
    def cleanup_states(self, data_dir: Path):
        """清理不存在文件的状态缓存"""
        existing_files = {str(f) for f in data_dir.glob("*.parquet")}
        
        # 移除不存在文件的状态
        self.file_states = {
            path: state for path, state in self.file_states.items() 
            if path in existing_files
        }

# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检测parquet文件完整性')
    parser.add_argument('--data-dir', required=True, help='数据目录')
    parser.add_argument('--stability-window', type=int, default=30, help='稳定性窗口(秒)')
    parser.add_argument('--watch', action='store_true', help='持续监控模式')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    detector = IncrementalFileDetector(args.stability_window)
    data_dir = Path(args.data_dir)
    
    if args.watch:
        print(f"🔍 持续监控: {data_dir}")
        while True:
            ready_files = detector.scan_ready_files(data_dir)
            print(f"📊 准备就绪: {len(ready_files)} 个文件")
            
            detector.cleanup_states(data_dir)
            time.sleep(10)
    else:
        ready_files = detector.scan_ready_files(data_dir)
        print(f"📋 可处理文件:")
        for f in ready_files:
            print(f"  {f}")
