#!/usr/bin/env python3
"""
FastText serving 客户端示例
用于大规模代码质量分类的数据清洗
"""

import argparse
import asyncio
import aiohttp
import json
import time
import pandas as pd
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class FastTextClient:
    def __init__(self, base_url: str = "http://localhost:8000", max_concurrent: int = 10):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        # 配置更大的连接池和超时
        connector = aiohttp.TCPConnector(
            limit=100,  # 总连接池大小
            limit_per_host=50,  # 每个主机的连接数
        )
        timeout = aiohttp.ClientTimeout(total=300)  # 5分钟超时
        self.session = aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def predict_batch(self, texts: List[str], k: int = 1, threshold: float = 0.0) -> List[Dict]:
        """批量预测"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' statement.")
            
        url = f"{self.base_url}/predict"
        params = {"k": k, "threshold": threshold}
        payload = texts
        
        try:
            async with self.session.post(url, json=payload, params=params) as response:
                if response.status == 200:
                    results = await response.json()
                    # 转换格式：[(labels, probs)] -> [{"labels": [], "scores": []}]
                    formatted_results = []
                    for labels, scores in results:
                        formatted_results.append({
                            "labels": labels,
                            "scores": scores,
                            "prediction": labels[0] if labels else "unknown",
                            "confidence": scores[0] if scores else 0.0
                        })
                    return formatted_results
                else:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {error_text}")
        except Exception as e:
            print(f"Prediction error: {e}")
            # 返回默认结果而不是崩溃
            return [{"labels": ["error"], "scores": [0.0], "prediction": "error", "confidence": 0.0} for _ in texts]


class DataProcessor:
    def __init__(self, client: FastTextClient, batch_size: int = 1000):
        self.client = client
        self.batch_size = batch_size
        
    def preprocess(self, text: str) -> str:
        """前处理：目前什么也不做"""
        # TODO: 这里可以添加文本清理、格式化等逻辑
        return text
        
    def postprocess(self, prediction_result: Dict[str, Any], original_data: Dict[str, Any]) -> Dict[str, Any]:
        """后处理：目前什么也不做，只是合并结果"""
        result = original_data.copy()
        result.update({
            "category_prediction": prediction_result["prediction"],
            "category_confidence": prediction_result["confidence"],
            "category_labels": prediction_result["labels"],
            "category_scores": prediction_result["scores"]
        })
        return result
        
    async def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理一个批次的数据"""
        # 前处理
        texts = [self.preprocess(item["text"]) for item in batch_data]
        
        # 推理
        predictions = await self.client.predict_batch(texts, k=2, threshold=0.1)
        
        # 后处理
        results = []
        for original_data, prediction in zip(batch_data, predictions):
            processed_result = self.postprocess(prediction, original_data)
            results.append(processed_result)
            
        return results
        
    async def process_data_parallel(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """并发处理数据"""
        print(f"Processing {len(data)} records in batches of {self.batch_size}")
        
        # 分批
        batches = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        print(f"Split into {len(batches)} batches")
        
        # 并发处理批次
        semaphore = asyncio.Semaphore(self.client.max_concurrent)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await self.process_batch(batch)
        
        start_time = time.time()
        batch_results = await asyncio.gather(
            *[process_with_semaphore(batch) for batch in batches],
            return_exceptions=True
        )
        end_time = time.time()
        
        # 合并结果，处理异常
        all_results = []
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                print(f"Batch {i} failed: {batch_result}")
                # 为失败的批次创建默认结果
                batch_data = batches[i]
                error_results = []
                for item in batch_data:
                    error_result = item.copy()
                    error_result.update({
                        "category_prediction": "error",
                        "category_confidence": 0.0,
                        "category_labels": ["error"],
                        "category_scores": [0.0]
                    })
                    error_results.append(error_result)
                all_results.extend(error_results)
            else:
                all_results.extend(batch_result)
        
        print(f"Processed {len(all_results)} records in {end_time - start_time:.2f} seconds")
        print(f"Throughput: {len(all_results)/(end_time - start_time):.2f} records/sec")
        
        return all_results


def load_test_data(use_test_data: bool = True, parquet_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载测试数据 - 使用cooking相关的测试数据"""
    if use_test_data:
        # 测试数据：烹饪问题分类场景
        test_data = [
            {
                "id": 1,
                "text": "Which baking dish is best to bake a banana bread?",
                "source": "cooking_forum",
                "author": "home_baker"
            },
            {
                "id": 2,
                "text": "Why not put knives in the dishwasher?",
                "source": "kitchen_tips",
                "author": "chef_mike"
            },
            {
                "id": 3,
                "text": "How do I make the perfect chocolate chip cookies?",
                "source": "recipe_blog",
                "author": "dessert_lover"
            },
            {
                "id": 4,
                "text": "What temperature should I use for roasting vegetables?",
                "source": "healthy_cooking",
                "author": "veggie_enthusiast"
            },
            {
                "id": 5,
                "text": "How long should I marinate chicken for grilling?",
                "source": "bbq_masters",
                "author": "grill_expert"
            },
            {
                "id": 6,
                "text": "What's the difference between baking soda and baking powder?",
                "source": "baking_science",
                "author": "pastry_chef"
            },
            {
                "id": 7,
                "text": "How can I prevent my pasta from sticking together?",
                "source": "italian_cuisine",
                "author": "pasta_lover"
            },
            {
                "id": 8,
                "text": "What knife should I use for chopping vegetables?",
                "source": "kitchen_equipment",
                "author": "culinary_student"
            },
            {
                "id": 9,
                "text": "How do I know when my steak is medium rare?",
                "source": "meat_cooking",
                "author": "steakhouse_chef"
            },
            {
                "id": 10,
                "text": "What oil is best for deep frying?",
                "source": "frying_techniques",
                "author": "restaurant_cook"
            },
            {
                "id": 11,
                "text": "How do I make homemade bread rise properly?",
                "source": "bread_making",
                "author": "home_baker_pro"
            },
            {
                "id": 12,
                "text": "What spices go well with lamb?",
                "source": "meat_seasoning",
                "author": "spice_expert"
            }
        ]
        
        # 复制数据以创建更大的测试集
        multiplier = 50  # 创建600个测试样本
        extended_data = []
        for i in range(multiplier):
            for j, item in enumerate(test_data):
                new_item = item.copy()
                new_item["id"] = i * len(test_data) + j + 1
                extended_data.append(new_item)
        
        print(f"Generated {len(extended_data)} cooking test samples")
        return extended_data
        
    else:
        # 从 parquet 文件加载
        if not parquet_path or not Path(parquet_path).exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} records from {parquet_path}")
        
        # 转换为字典列表
        return df.to_dict('records')


def save_results(results: List[Dict[str, Any]], output_path: str):
    """保存结果到文件"""
    output_path = Path(output_path)
    
    if output_path.suffix.lower() == '.parquet':
        # 保存为 parquet
        df = pd.DataFrame(results)
        df.to_parquet(output_path, index=False)
        print(f"Results saved to {output_path} (parquet format)")
    else:
        # 保存为 JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_path} (json format)")


async def main():
    parser = argparse.ArgumentParser(description="FastText serving client for cooking question classification")
    parser.add_argument("--url", default="http://localhost:8000", help="FastText serving URL")
    parser.add_argument("--input", help="Input parquet file path")
    parser.add_argument("--output", default="cooking_results.json", help="Output file path")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent requests")
    parser.add_argument("--use-test-data", action="store_true", help="Use test data instead of file")
    
    args = parser.parse_args()
    
    # 加载数据
    try:
        data = load_test_data(use_test_data=args.use_test_data, parquet_path=args.input)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # 处理数据
    async with FastTextClient(args.url, args.max_concurrent) as client:
        processor = DataProcessor(client, args.batch_size)
        
        try:
            results = await processor.process_data_parallel(data)
            
            # 保存结果
            save_results(results, args.output)
            
            # 打印统计信息
            category_stats = {}
            for result in results:
                prediction = result.get("category_prediction", "unknown")
                category_stats[prediction] = category_stats.get(prediction, 0) + 1
            
            print("\n=== Cooking Category Classification Results ===")
            for category, count in sorted(category_stats.items()):
                percentage = (count / len(results)) * 100
                print(f"{category}: {count} ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"Processing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 