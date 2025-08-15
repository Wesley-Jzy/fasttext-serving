#!/usr/bin/env python3
"""
系统资源监控工具
监控CPU、内存、网络使用情况，帮助调试性能问题
"""

import psutil
import time
import argparse
from typing import Dict, List

class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.history: List[Dict] = []
    
    def get_current_stats(self) -> Dict:
        """获取当前系统状态"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 网络使用情况
        net_io = psutil.net_io_counters()
        
        # 磁盘使用情况
        disk_io = psutil.disk_io_counters()
        
        # 进程信息
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['cpu_percent'] > 1.0 or proc.info['memory_percent'] > 1.0:
                    processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # 按CPU使用率排序
        processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
        
        return {
            'timestamp': time.time(),
            'cpu': {
                'total_percent': cpu_percent,
                'count': cpu_count,
                'per_core': cpu_per_core
            },
            'memory': {
                'total_gb': memory.total / (1024**3),
                'used_gb': memory.used / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent': memory.percent
            },
            'swap': {
                'total_gb': swap.total / (1024**3),
                'used_gb': swap.used / (1024**3),
                'percent': swap.percent
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'disk': {
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0
            },
            'top_processes': processes[:10]  # 前10个高负载进程
        }
    
    def print_stats(self, stats: Dict, previous_stats: Dict = None):
        """打印统计信息"""
        print(f"\n{'='*60}")
        print(f"⏰ 时间: {time.strftime('%H:%M:%S', time.localtime(stats['timestamp']))}")
        
        # CPU信息
        print(f"\n🖥️  CPU使用率:")
        print(f"  总体: {stats['cpu']['total_percent']:.1f}% ({stats['cpu']['count']} 核)")
        
        # 显示前8个核心的使用率
        cores_to_show = min(8, len(stats['cpu']['per_core']))
        core_usage = ' '.join([f"C{i}:{stats['cpu']['per_core'][i]:.0f}%" 
                              for i in range(cores_to_show)])
        print(f"  分核: {core_usage}")
        if len(stats['cpu']['per_core']) > 8:
            print(f"  ... 还有 {len(stats['cpu']['per_core']) - 8} 个核心")
        
        # 内存信息
        mem = stats['memory']
        print(f"\n💾 内存使用:")
        print(f"  总计: {mem['total_gb']:.1f} GB")
        print(f"  已用: {mem['used_gb']:.1f} GB ({mem['percent']:.1f}%)")
        print(f"  可用: {mem['available_gb']:.1f} GB")
        
        # 交换空间
        if stats['swap']['total_gb'] > 0:
            swap = stats['swap']
            print(f"  交换: {swap['used_gb']:.1f}/{swap['total_gb']:.1f} GB ({swap['percent']:.1f}%)")
        
        # 网络信息
        if previous_stats:
            time_diff = stats['timestamp'] - previous_stats['timestamp']
            if time_diff > 0:
                net_sent_rate = (stats['network']['bytes_sent'] - previous_stats['network']['bytes_sent']) / time_diff
                net_recv_rate = (stats['network']['bytes_recv'] - previous_stats['network']['bytes_recv']) / time_diff
                
                print(f"\n🌐 网络速率:")
                print(f"  上传: {net_sent_rate / (1024**2):.1f} MB/s")
                print(f"  下载: {net_recv_rate / (1024**2):.1f} MB/s")
        
        # 高负载进程
        if stats['top_processes']:
            print(f"\n🔥 高负载进程 (前5个):")
            print(f"{'PID':<8} {'名称':<20} {'CPU%':<8} {'内存%':<8}")
            print("-" * 50)
            for proc in stats['top_processes'][:5]:
                print(f"{proc['pid']:<8} {proc['name'][:19]:<20} {proc['cpu_percent']:<8.1f} {proc['memory_percent']:<8.1f}")
    
    def check_system_health(self, stats: Dict) -> List[str]:
        """检查系统健康状况"""
        warnings = []
        
        # CPU检查
        if stats['cpu']['total_percent'] > 90:
            warnings.append("⚠️ CPU使用率超过90%")
        elif stats['cpu']['total_percent'] > 70:
            warnings.append("⚠️ CPU使用率较高 (>70%)")
        
        # 内存检查
        if stats['memory']['percent'] > 90:
            warnings.append("⚠️ 内存使用率超过90%")
        elif stats['memory']['percent'] > 80:
            warnings.append("⚠️ 内存使用率较高 (>80%)")
        
        # 交换空间检查
        if stats['swap']['total_gb'] > 0 and stats['swap']['percent'] > 50:
            warnings.append("⚠️ 交换空间使用率超过50%")
        
        return warnings
    
    def monitor(self, duration: int = None):
        """开始监控"""
        print(f"🚀 开始系统监控 (每{self.interval}秒刷新)")
        if duration:
            print(f"📅 监控时长: {duration}秒")
        print("按 Ctrl+C 停止监控")
        
        start_time = time.time()
        previous_stats = None
        
        try:
            while True:
                current_stats = self.get_current_stats()
                self.history.append(current_stats)
                
                # 清屏 (在支持的终端中)
                import os
                if os.name == 'nt':  # Windows
                    os.system('cls')
                else:  # Unix/Linux
                    os.system('clear')
                
                self.print_stats(current_stats, previous_stats)
                
                # 健康检查
                warnings = self.check_system_health(current_stats)
                if warnings:
                    print(f"\n🚨 系统警告:")
                    for warning in warnings:
                        print(f"  {warning}")
                
                previous_stats = current_stats
                
                # 检查是否达到监控时长
                if duration and (time.time() - start_time) >= duration:
                    break
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print(f"\n\n✅ 监控已停止")
            self.print_summary()
    
    def print_summary(self):
        """打印监控摘要"""
        if not self.history:
            return
        
        print(f"\n📊 监控摘要:")
        print(f"监控时长: {len(self.history) * self.interval} 秒")
        
        # CPU平均使用率
        avg_cpu = sum(s['cpu']['total_percent'] for s in self.history) / len(self.history)
        max_cpu = max(s['cpu']['total_percent'] for s in self.history)
        print(f"CPU: 平均 {avg_cpu:.1f}%, 峰值 {max_cpu:.1f}%")
        
        # 内存平均使用率
        avg_mem = sum(s['memory']['percent'] for s in self.history) / len(self.history)
        max_mem = max(s['memory']['percent'] for s in self.history)
        print(f"内存: 平均 {avg_mem:.1f}%, 峰值 {max_mem:.1f}%")

def main():
    parser = argparse.ArgumentParser(description='系统资源监控工具')
    parser.add_argument('--interval', type=int, default=5, help='监控间隔(秒) (默认: 5)')
    parser.add_argument('--duration', type=int, help='监控时长(秒) (默认: 无限制)')
    
    args = parser.parse_args()
    
    monitor = SystemMonitor(interval=args.interval)
    monitor.monitor(duration=args.duration)

if __name__ == "__main__":
    main()
