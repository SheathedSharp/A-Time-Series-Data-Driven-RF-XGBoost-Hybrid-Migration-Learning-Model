"""
故障分析脚本
分析CSV文件中各个故障发生的具体数据，包括开始时间、结束时间、持续时间和频率
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class FaultAnalyzer:
    def __init__(self, csv_file_path: str):
        """
        初始化故障分析器
        
        Args:
            csv_file_path: CSV文件路径
        """
        self.csv_file_path = Path(csv_file_path)
        self.data = None
        self.fault_columns = []
        
    def load_data(self):
        """加载CSV数据"""
        print(f"正在加载数据文件: {self.csv_file_path}")
        try:
            # 分块读取大文件
            chunk_size = 10000
            chunks = []
            for chunk in pd.read_csv(self.csv_file_path, chunksize=chunk_size):
                chunks.append(chunk)
            self.data = pd.concat(chunks, ignore_index=True)
            
            # 创建完整的时间戳
            # Date是Julian日期（1=第1天），Time是秒数计数器
            # 将Julian日期转换为实际日期，然后加上秒数
            base_date = pd.to_datetime('2024-01-01')  # 假设数据从2024年开始
            self.data['datetime'] = (
                base_date + 
                pd.to_timedelta(self.data['Date'] - 1, unit='D') + 
                pd.to_timedelta(self.data['Time'], unit='s')
            )
            
            # 识别故障列（包含"Fault"的列）
            self.fault_columns = [col for col in self.data.columns if 'Fault' in col]
            print(f"识别到 {len(self.fault_columns)} 个故障类型:")
            for i, fault in enumerate(self.fault_columns, 1):
                print(f"  {i}. {fault}")
                
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise
    
    def analyze_fault_events(self, fault_column: str) -> List[Dict]:
        """
        分析单个故障的事件
        
        Args:
            fault_column: 故障列名
            
        Returns:
            故障事件列表，每个事件包含开始时间、结束时间、持续时间等信息
        """
        if fault_column not in self.data.columns:
            print(f"警告: 未找到故障列 '{fault_column}'")
            return []
        
        fault_series = self.data[fault_column]
        events = []
        
        # 将故障值转换为二进制（0=正常，>0=故障）
        fault_binary = (fault_series > 0).astype(int)
        
        # 找到故障状态变化的位置
        fault_changes = fault_binary.diff().fillna(0)
        
        # 找到故障开始的位置（从0变为1）
        fault_starts = self.data.index[fault_changes == 1].tolist()
        
        # 找到故障结束的位置（从1变为0）
        fault_ends = self.data.index[fault_changes == -1].tolist()
        
        # 如果数据开始时就是故障状态，在开头添加一个开始点
        if len(fault_binary) > 0 and fault_binary.iloc[0] == 1:
            fault_starts.insert(0, 0)
        
        # 如果数据结束时仍是故障状态，在末尾添加一个结束点
        if len(fault_binary) > 0 and fault_binary.iloc[-1] == 1:
            fault_ends.append(len(fault_binary) - 1)
        
        # 匹配开始和结束时间
        for i, start_idx in enumerate(fault_starts):
            if i < len(fault_ends):
                end_idx = fault_ends[i]
                
                start_time = self.data.iloc[start_idx]['datetime']
                end_time = self.data.iloc[end_idx]['datetime']
                duration = end_time - start_time
                
                events.append({
                    'fault_type': fault_column,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_seconds': duration.total_seconds(),
                    'duration_minutes': duration.total_seconds() / 60,
                    'start_index': start_idx,
                    'end_index': end_idx
                })
        
        return events
    
    def analyze_all_faults(self) -> Dict[str, List[Dict]]:
        """分析所有故障类型"""
        all_fault_events = {}
        
        for fault_column in self.fault_columns:
            print(f"\n正在分析故障: {fault_column}")
            events = self.analyze_fault_events(fault_column)
            all_fault_events[fault_column] = events
            print(f"  发现 {len(events)} 次故障事件")
        
        return all_fault_events
    
    def generate_fault_summary(self, fault_events: Dict[str, List[Dict]]) -> pd.DataFrame:
        """生成故障摘要报告"""
        summary_data = []
        
        for fault_type, events in fault_events.items():
            if events:
                durations = [event['duration_minutes'] for event in events]
                
                summary_data.append({
                    '故障类型': fault_type,
                    '发生次数': len(events),
                    '总持续时间(分钟)': sum(durations),
                    '平均持续时间(分钟)': np.mean(durations),
                    '最短持续时间(分钟)': min(durations),
                    '最长持续时间(分钟)': max(durations),
                    '标准差(分钟)': np.std(durations)
                })
            else:
                summary_data.append({
                    '故障类型': fault_type,
                    '发生次数': 0,
                    '总持续时间(分钟)': 0,
                    '平均持续时间(分钟)': 0,
                    '最短持续时间(分钟)': 0,
                    '最长持续时间(分钟)': 0,
                    '标准差(分钟)': 0
                })
        
        return pd.DataFrame(summary_data)
    
    def save_detailed_report(self, fault_events: Dict[str, List[Dict]], output_dir: str = "experiments/results/fault_analysis_results"):
        """保存详细的故障分析报告"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存每个故障类型的详细事件
        for fault_type, events in fault_events.items():
            if events:
                df = pd.DataFrame(events)
                # 清理文件名
                safe_filename = fault_type.replace(' ', '_').replace('/', '_')
                df.to_csv(output_path / f"{safe_filename}_详细事件.csv", index=False, encoding='utf-8-sig')
        
        # 保存汇总报告
        summary_df = self.generate_fault_summary(fault_events)
        summary_df.to_csv(output_path / "故障汇总报告.csv", index=False, encoding='utf-8-sig')
        
        print(f"\n详细报告已保存到: {output_path}")
        return summary_df
    
    def create_visualizations(self, fault_events: Dict[str, List[Dict]], output_dir: str = "experiments/results/fault_analysis_results"):
        """创建可视化图表"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 故障频率柱状图
        fault_counts = {fault: len(events) for fault, events in fault_events.items() if events}
        if fault_counts:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(range(len(fault_counts)), list(fault_counts.values()))
            plt.xlabel('故障类型')
            plt.ylabel('发生次数')
            plt.title('各故障类型发生频率')
            plt.xticks(range(len(fault_counts)), list(fault_counts.keys()), rotation=45, ha='right')
            
            # 在柱子上显示数值
            for bar, count in zip(bars, fault_counts.values()):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_path / "故障频率分布.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 故障持续时间分布（针对有事件的故障）
        for fault_type, events in fault_events.items():
            if len(events) > 1:  # 只为有多个事件的故障创建分布图
                durations = [event['duration_minutes'] for event in events]
                
                plt.figure(figsize=(10, 6))
                plt.hist(durations, bins=20, alpha=0.7, edgecolor='black')
                plt.xlabel('持续时间 (分钟)')
                plt.ylabel('频次')
                plt.title(f'{fault_type} - 故障持续时间分布')
                plt.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_duration = np.mean(durations)
                plt.axvline(mean_duration, color='red', linestyle='--', 
                           label=f'平均值: {mean_duration:.1f} 分钟')
                plt.legend()
                
                safe_filename = fault_type.replace(' ', '_').replace('/', '_')
                plt.tight_layout()
                plt.savefig(output_path / f"{safe_filename}_持续时间分布.png", dpi=300, bbox_inches='tight')
                plt.close()
    
    def analyze_1001_fault_specifically(self) -> Dict:
        """专门分析1001故障"""
        fault_1001_column = None
        for col in self.fault_columns:
            if '1001' in col:
                fault_1001_column = col
                break
        
        if not fault_1001_column:
            print("未找到1001故障列")
            return {}
        
        print(f"\n=== 专门分析1001故障: {fault_1001_column} ===")
        events = self.analyze_fault_events(fault_1001_column)
        
        if not events:
            print("未发现1001故障事件")
            return {}
        
        print(f"1001故障总计发生 {len(events)} 次")
        print("\n详细事件列表:")
        print("-" * 80)
        
        total_duration = 0
        for i, event in enumerate(events, 1):
            print(f"事件 {i}:")
            print(f"  开始时间: {event['start_time']}")
            print(f"  结束时间: {event['end_time']}")
            print(f"  持续时间: {event['duration_minutes']:.2f} 分钟")
            print()
            total_duration += event['duration_minutes']
        
        print(f"1001故障统计摘要:")
        print(f"  总发生次数: {len(events)}")
        print(f"  总持续时间: {total_duration:.2f} 分钟")
        print(f"  平均持续时间: {total_duration/len(events):.2f} 分钟")
        
        durations = [event['duration_minutes'] for event in events]
        print(f"  最短持续时间: {min(durations):.2f} 分钟")
        print(f"  最长持续时间: {max(durations):.2f} 分钟")
        print(f"  标准差: {np.std(durations):.2f} 分钟")
        
        return {
            'events': events,
            'summary': {
                'total_events': len(events),
                'total_duration_minutes': total_duration,
                'average_duration_minutes': total_duration/len(events),
                'min_duration_minutes': min(durations),
                'max_duration_minutes': max(durations),
                'std_duration_minutes': np.std(durations)
            }
        }

def main():
    parser = argparse.ArgumentParser(description='故障数据分析工具')
    parser.add_argument('--input', '-i', default='data/raw/production_line_1.csv',
                       help='输入CSV文件路径 (默认: data/raw/production_line_1.csv)')
    parser.add_argument('--output', '-o', default='experiments/results/fault_analysis_results',
                       help='输出目录 (默认: experiments/results/fault_analysis_results)')
    parser.add_argument('--focus-1001', action='store_true',
                       help='重点分析1001故障')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = FaultAnalyzer(args.input)
    
    try:
        # 加载数据
        analyzer.load_data()
        
        # 分析所有故障
        fault_events = analyzer.analyze_all_faults()
        
        # 生成报告
        summary_df = analyzer.save_detailed_report(fault_events, args.output)
        print("\n=== 故障汇总报告 ===")
        print(summary_df.to_string(index=False))
        
        # 创建可视化
        analyzer.create_visualizations(fault_events, args.output)
        
        # 如果指定了重点分析1001故障
        if args.focus_1001:
            analyzer.analyze_1001_fault_specifically()
        
        print(f"\n分析完成！所有结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 