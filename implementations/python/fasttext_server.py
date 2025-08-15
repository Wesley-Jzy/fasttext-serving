#!/usr/bin/env python3
"""
FastText Serving - Python Implementation

使用官方fasttext v0.9.2库，提供与Rust版本完全一致的HTTP API
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
import json

import fasttext
import numpy as np
from flask import Flask, request, jsonify, Response


class FastTextServer:
    """FastText服务器类"""
    
    def __init__(self, 
                 model_path: str,
                 max_text_length: int = 10_000_000,
                 default_threshold: float = 0.0,
                 default_vector_dim: int = 100):
        self.model_path = model_path
        self.max_text_length = max_text_length
        self.default_threshold = default_threshold
        self.default_vector_dim = default_vector_dim
        self.model = None
        self.model_loaded = False
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载FastText模型"""
        try:
            self.logger.info(f"Loading FastText model from: {self.model_path}")
            start_time = time.time()
            
            self.model = fasttext.load_model(self.model_path)
            self.model_loaded = True
            
            load_time = time.time() - start_time
            self.logger.info(f"FastText model loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to load FastText model: {e}")
            raise
    
    def predict_single(self, 
                      text: str, 
                      k: int = 1, 
                      threshold: float = None) -> Tuple[List[str], List[float]]:
        """
        单个文本预测
        
        Args:
            text: 输入文本
            k: 返回top-k标签
            threshold: 预测阈值
            
        Returns:
            (labels, scores): 标签列表和分数列表
        """
        # 参数验证
        if not text or not text.strip():
            raise ValueError("Empty text input")
        
        if len(text) > self.max_text_length:
            raise ValueError(f"Text too long: {len(text)} bytes")
        
        if threshold is None:
            threshold = self.default_threshold
            
        k = max(1, k)  # 确保k >= 1
        
        try:
            # 使用官方FastText预测
            labels, scores = self.model.predict(text, k=k, threshold=threshold)
            
            # 转换为Python原生类型，保持完整的__label__格式
            labels_list = list(labels)  # 保持原始标签格式
            scores_list = scores.tolist()
            
            return labels_list, scores_list
            
        except Exception as e:
            self.logger.error(f"Prediction failed for text (length: {len(text)}): {e}")
            # 返回默认结果
            return ["__label__error"], [0.0]
    
    def predict_batch(self, 
                     texts: List[str], 
                     k: int = 1, 
                     threshold: float = None) -> List[Dict[str, Any]]:
        """
        批量文本预测
        
        Args:
            texts: 文本列表
            k: 返回top-k标签
            threshold: 预测阈值
            
        Returns:
            预测结果列表，格式: [{"labels": [...], "scores": [...]}, ...]
        """
        if not texts:
            return []
        
        results = []
        success_count = 0
        error_count = 0
        
        for text in texts:
            try:
                labels, scores = self.predict_single(text, k, threshold)
                results.append({
                    "labels": labels,
                    "scores": scores
                })
                success_count += 1
                
            except Exception as e:
                self.logger.warning(f"Prediction failed for text (length: {len(text) if text else 0}): {e}")
                # 返回错误结果，不中断整个批次
                results.append({
                    "labels": ["__label__error"],
                    "scores": [0.0]
                })
                error_count += 1
        
        # 记录批次处理结果
        total_texts = len(texts)
        if error_count > 0:
            self.logger.warning(f"Batch processing completed with {error_count} errors out of {total_texts} texts")
        else:
            self.logger.info(f"Batch processing completed successfully: {success_count} texts")
        
        return results
    
    def get_sentence_vectors(self, texts: List[str]) -> List[List[float]]:
        """
        获取句向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        results = []
        
        for text in texts:
            try:
                if not text or not text.strip():
                    # 空文本返回零向量
                    results.append([0.0] * self.default_vector_dim)
                    continue
                
                vector = self.model.get_sentence_vector(text)
                results.append(vector.tolist())
                
            except Exception as e:
                self.logger.warning(f"Sentence vector failed for text (length: {len(text) if text else 0}): {e}")
                # 返回零向量
                results.append([0.0] * self.default_vector_dim)
        
        return results


def create_app(server: FastTextServer) -> Flask:
    """创建Flask应用"""
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health():
        """健康检查"""
        return jsonify({
            "status": "healthy",
            "model_loaded": server.model_loaded,
            "implementation": "python",
            "version": "1.0.0",
            "model_path": server.model_path
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """预测接口"""
        try:
            # 解析请求
            if not request.is_json:
                return jsonify({"error": "Bad Request", "message": "Content-Type must be application/json"}), 400
            
            texts = request.get_json()
            if not isinstance(texts, list):
                return jsonify({"error": "Bad Request", "message": "Request body must be a list of strings"}), 400
            
            # 解析参数
            k = request.args.get('k', 1, type=int)
            threshold = request.args.get('threshold', server.default_threshold, type=float)
            
            server.logger.info(f"Processing {len(texts)} texts with k={k}, threshold={threshold}")
            
            # 执行预测
            start_time = time.time()
            results = server.predict_batch(texts, k, threshold)
            processing_time = time.time() - start_time
            
            server.logger.info(f"Prediction completed in {processing_time:.3f}s")
            
            return jsonify(results)
            
        except Exception as e:
            server.logger.error(f"Prediction error: {e}")
            return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
    @app.route('/sentence-vector', methods=['POST'])
    def sentence_vector():
        """句向量接口"""
        try:
            if not request.is_json:
                return jsonify({"error": "Bad Request", "message": "Content-Type must be application/json"}), 400
            
            texts = request.get_json()
            if not isinstance(texts, list):
                return jsonify({"error": "Bad Request", "message": "Request body must be a list of strings"}), 400
            
            server.logger.info(f"Processing {len(texts)} texts for sentence vectors")
            
            # 获取句向量
            start_time = time.time()
            results = server.get_sentence_vectors(texts)
            processing_time = time.time() - start_time
            
            server.logger.info(f"Sentence vector generation completed in {processing_time:.3f}s")
            
            return jsonify(results)
            
        except Exception as e:
            server.logger.error(f"Sentence vector error: {e}")
            return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({"error": "Request Entity Too Large", "message": "Request body too large"}), 413
    
    return app


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='FastText Serving - Python Implementation')
    parser.add_argument('--model', '-m', required=True, help='FastText model path')
    parser.add_argument('--address', '-a', default='0.0.0.0', help='Listen address')
    parser.add_argument('--port', '-p', type=int, default=8000, help='Listen port')
    parser.add_argument('--max-text-length', type=int, default=10_000_000, 
                       help='Maximum text length in bytes')
    parser.add_argument('--default-threshold', type=float, default=0.0,
                       help='Default prediction threshold')
    parser.add_argument('--default-vector-dim', type=int, default=100,
                       help='Default vector dimension for errors')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # 验证模型文件
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        sys.exit(1)
    
    try:
        # 创建服务器
        server = FastTextServer(
            model_path=args.model,
            max_text_length=args.max_text_length,
            default_threshold=args.default_threshold,
            default_vector_dim=args.default_vector_dim
        )
        
        # 创建Flask应用
        app = create_app(server)
        
        # 启动服务
        print(f"Starting FastText server (Python) on {args.address}:{args.port}")
        print(f"Model: {args.model}")
        # 尝试获取FastText版本号
        try:
            ft_version = getattr(fasttext, '__version__', 'unknown')
        except:
            ft_version = 'unknown'
        print(f"Implementation: Python with fasttext v{ft_version}")
        
        app.run(
            host=args.address,
            port=args.port,
            debug=args.debug,
            threaded=True  # 启用多线程支持
        )
        
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
