# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
Track profiler module for measuring and logging timing information in tracking pipeline.
Generates CSV files for performance analysis.
"""

import csv
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from ultralytics.utils import LOGGER


class TrackProfiler:
    """Singleton profiler for tracking pipeline performance measurement.
    
    This class collects timing data from various tracking components and exports
    to CSV file for analysis. Thread-safe for multi-threaded preprocessing.
    
    Attributes:
        enabled (bool): Whether profiling is enabled.
        records (list): List of timing records per frame.
        current_frame (dict): Current frame's timing data.
        output_dir (Path): Directory to save CSV files.
        
    Examples:
        Enable profiling and measure timing
        >>> profiler = TrackProfiler.get_instance()
        >>> profiler.enable(output_dir="./runs")
        >>> with profiler.measure("reid_preprocess"):
        ...     # preprocessing code
        >>> profiler.end_frame()
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.enabled = False
        self.records = []
        self.current_frame = defaultdict(float)
        self.frame_id = 0
        self.output_dir = None
        self.start_time = None
        self._measure_lock = Lock()
        
        # Define timing categories for consistent ordering
        self.timing_categories = [
            "frame_id",
            "num_detections",
            "tracker_total",
            "reid_total",
            "reid_preprocess",
            "reid_inference", 
            "init_track",
            "kalman_predict",
            "gmc",
            "get_dists",
            "iou_distance",
            "embedding_distance",
            "linear_assignment",
            "track_update",
        ]
    
    @classmethod
    def get_instance(cls) -> "TrackProfiler":
        """Get the singleton profiler instance."""
        return cls()
    
    def enable(self, output_dir: Optional[str] = None) -> None:
        """Enable profiling and set output directory.
        
        Args:
            output_dir (str, optional): Directory to save CSV files. 
                                        Defaults to current working directory.
        """
        self.enabled = True
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = datetime.now()
        self.records = []
        self.frame_id = 0
        LOGGER.info(f"📊 Track profiler enabled, output dir: {self.output_dir}")
    
    def disable(self) -> None:
        """Disable profiling."""
        self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self.enabled
    
    @contextmanager
    def measure(self, name: str):
        """Context manager to measure execution time of a code block.
        
        Args:
            name (str): Name of the timing category.
            
        Yields:
            None
            
        Examples:
            >>> with profiler.measure("reid_inference"):
            ...     outputs = model(inputs)
        """
        if not self.enabled:
            yield
            return
            
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
            with self._measure_lock:
                self.current_frame[name] += elapsed
    
    def record(self, name: str, value: float) -> None:
        """Record a timing or metric value directly.
        
        Args:
            name (str): Name of the metric.
            value (float): Value to record (in ms for timing).
        """
        if not self.enabled:
            return
        with self._measure_lock:
            self.current_frame[name] = value
    
    def add_time(self, name: str, elapsed_ms: float) -> None:
        """Add elapsed time to a timing category (for accumulating).
        
        Args:
            name (str): Name of the timing category.
            elapsed_ms (float): Elapsed time in milliseconds.
        """
        if not self.enabled:
            return
        with self._measure_lock:
            self.current_frame[name] += elapsed_ms
    
    def start_frame(self, frame_id: int, num_detections: int = 0) -> None:
        """Start timing for a new frame.
        
        Args:
            frame_id (int): Current frame ID.
            num_detections (int): Number of detections in this frame.
        """
        if not self.enabled:
            return
        self.frame_id = frame_id
        self.current_frame = defaultdict(float)
        self.current_frame["frame_id"] = frame_id
        self.current_frame["num_detections"] = num_detections
    
    def end_frame(self) -> None:
        """End timing for current frame and save record."""
        if not self.enabled:
            return
        with self._measure_lock:
            self.records.append(dict(self.current_frame))
            self.current_frame = defaultdict(float)
    
    def save_csv(self, filename: Optional[str] = None) -> str:
        """Save timing records to CSV file.
        
        Args:
            filename (str, optional): Custom filename. Defaults to auto-generated name.
            
        Returns:
            (str): Path to the saved CSV file.
        """
        if not self.records:
            LOGGER.warning("No timing records to save")
            return ""
        
        if filename is None:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S") if self.start_time else datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"track_profile_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        # Collect all unique keys from records
        all_keys = set()
        for record in self.records:
            all_keys.update(record.keys())
        
        # Order columns: predefined categories first, then any additional ones
        columns = []
        for cat in self.timing_categories:
            if cat in all_keys:
                columns.append(cat)
                all_keys.discard(cat)
        columns.extend(sorted(all_keys))
        
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for record in self.records:
                # Fill missing values with 0
                row = {col: record.get(col, 0) for col in columns}
                writer.writerow(row)
        
        LOGGER.info(f"📊 Saved {len(self.records)} timing records to {filepath}")
        return str(filepath)
    
    def get_summary(self) -> dict:
        """Get summary statistics of timing data.
        
        Returns:
            (dict): Dictionary with mean, min, max, std for each timing category.
        """
        if not self.records:
            return {}
        
        import numpy as np
        
        summary = {}
        all_keys = set()
        for record in self.records:
            all_keys.update(record.keys())
        
        for key in all_keys:
            if key in ["frame_id"]:
                continue
            values = [r.get(key, 0) for r in self.records]
            summary[key] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "std": float(np.std(values)),
            }
        
        return summary
    
    def print_summary(self) -> None:
        """Print a formatted summary of timing statistics."""
        summary = self.get_summary()
        if not summary:
            LOGGER.info("No timing data to summarize")
            return
        
        LOGGER.info("\n" + "=" * 70)
        LOGGER.info("📊 TRACK PROFILER SUMMARY")
        LOGGER.info("=" * 70)
        LOGGER.info(f"{'Category':<25} {'Mean':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
        LOGGER.info("-" * 70)
        
        for cat in self.timing_categories:
            if cat in summary and cat not in ["frame_id", "num_detections"]:
                s = summary[cat]
                LOGGER.info(f"{cat:<25} {s['mean']:>9.2f}ms {s['min']:>9.2f}ms {s['max']:>9.2f}ms {s['std']:>9.2f}ms")
        
        # Print any additional categories not in predefined list
        for key, s in summary.items():
            if key not in self.timing_categories and key not in ["frame_id", "num_detections"]:
                LOGGER.info(f"{key:<25} {s['mean']:>9.2f}ms {s['min']:>9.2f}ms {s['max']:>9.2f}ms {s['std']:>9.2f}ms")
        
        LOGGER.info("=" * 70 + "\n")
    
    def reset(self) -> None:
        """Reset all timing records."""
        self.records = []
        self.current_frame = defaultdict(float)
        self.frame_id = 0


# Global profiler instance for easy access
_profiler = TrackProfiler.get_instance()


def get_profiler() -> TrackProfiler:
    """Get the global track profiler instance.
    
    Returns:
        (TrackProfiler): The singleton profiler instance.
    """
    return _profiler


def enable_profiling(output_dir: Optional[str] = None) -> None:
    """Enable track profiling globally.
    
    Args:
        output_dir (str, optional): Directory to save CSV files.
    """
    _profiler.enable(output_dir)


def disable_profiling() -> None:
    """Disable track profiling globally."""
    _profiler.disable()
