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
        
        # Buffered writing to avoid per-frame file I/O
        self._buffer = []
        self._buffer_size = 1000  # Flush every 1000 frames to reduce I/O blocking
        self._csv_file = None
        self._csv_handle = None
        self._csv_writer = None
        self._header_written = False
        
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
        
        # Initialize CSV file for buffered writing
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self._csv_file = self.output_dir / f"track_profile_{timestamp}.csv"
        self._buffer = []
        self._header_written = False
        self._csv_handle = None
        self._csv_writer = None
        
        LOGGER.info(f"📊 Track profiler enabled, output: {self._csv_file}")
    
    def disable(self) -> None:
        """Disable profiling and flush remaining data."""
        self._flush_buffer()
        self._close_csv()
        self.enabled = False
    
    def _open_csv(self) -> None:
        """Open CSV file for writing (called once on first flush)."""
        if self._csv_handle is None and self._csv_file is not None:
            self._csv_handle = open(self._csv_file, "w", newline="", buffering=8192)
    
    def _close_csv(self) -> None:
        """Close CSV file handle."""
        if self._csv_handle is not None:
            try:
                self._csv_handle.flush()
                self._csv_handle.close()
            except Exception:
                pass
            finally:
                self._csv_handle = None
                self._csv_writer = None
    
    def _flush_buffer(self) -> None:
        """Flush buffered records to CSV file."""
        if not self._buffer:
            return
        
        self._open_csv()
        if self._csv_handle is None:
            return
        
        # Collect all keys for columns
        all_keys = set()
        for record in self._buffer:
            all_keys.update(record.keys())
        
        # Order columns
        columns = []
        for cat in self.timing_categories:
            if cat in all_keys:
                columns.append(cat)
                all_keys.discard(cat)
        columns.extend(sorted(all_keys))
        
        # Initialize writer if needed (first flush or after reset)
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_handle, fieldnames=columns)
        
        # Write header on first flush
        if not self._header_written:
            self._csv_writer.writeheader()
            self._header_written = True
        
        # Write all buffered rows
        for record in self._buffer:
            row = {col: record.get(col, 0) for col in columns}
            self._csv_writer.writerow(row)
        
        self._buffer = []
    
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
        """End timing for current frame and buffer record."""
        if not self.enabled:
            return
        with self._measure_lock:
            record = dict(self.current_frame)
            self.records.append(record)  # Keep for summary stats
            self._buffer.append(record)  # Buffer for CSV writing
            self.current_frame = defaultdict(float)
            
            # Flush buffer when full
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()
    
    def save_csv(self, filename: Optional[str] = None) -> str:
        """Flush remaining buffer and finalize CSV file.
        
        Args:
            filename (str, optional): Ignored (file already created on enable).
            
        Returns:
            (str): Path to the saved CSV file.
        """
        # Flush any remaining buffered data
        self._flush_buffer()
        self._close_csv()
        
        if self._csv_file and self._csv_file.exists():
            LOGGER.info(f"📊 Saved {len(self.records)} timing records to {self._csv_file}")
            return str(self._csv_file)
        
        LOGGER.warning("No timing records to save")
        return ""
    
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
        """Reset all timing records and close files."""
        self._flush_buffer()
        self._close_csv()
        self.records = []
        self._buffer = []
        self.current_frame = defaultdict(float)
        self.frame_id = 0
        self._header_written = False


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
