from datetime import datetime
from typing import Dict, Any
import logging
import time

class PipelineMonitor:
    """Monitor pipeline performance and stats"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {
            "processed": 0,
            "failed": 0,
            "skipped": 0
        }
        
    def start_pipeline(self):
        """Start pipeline monitoring"""
        self.start_time = time.time()
        self.metrics = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "by_category": {}
        }
        
    def record_success(self, category: str):
        """Record successful processing"""
        self.metrics["processed"] += 1
        if category not in self.metrics["by_category"]:
            self.metrics["by_category"][category] = {"processed": 0, "failed": 0}
        self.metrics["by_category"][category]["processed"] += 1
        
    def record_failure(self, category: str):
        """Record processing failure"""
        self.metrics["failed"] += 1
        if category not in self.metrics["by_category"]:
            self.metrics["by_category"][category] = {"processed": 0, "failed": 0}
        self.metrics["by_category"][category]["failed"] += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        duration = time.time() - self.start_time if self.start_time else 0
        
        return {
            "duration_seconds": round(duration, 2),
            "processed_count": self.metrics["processed"],
            "failed_count": self.metrics["failed"],
            "success_rate": round(
                self.metrics["processed"] / 
                (self.metrics["processed"] + self.metrics["failed"]) * 100, 2
            ) if self.metrics["processed"] > 0 else 0,
            "by_category": self.metrics["by_category"],
            "timestamp": datetime.now()
        }
