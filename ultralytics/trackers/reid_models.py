# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""
ReID models for object tracking with TensorRT support.
Supports YOLO, OSNet (PyTorch/ONNX/TensorRT), and custom ReID models.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box


class BaseReIDEncoder:
    """Base class for ReID encoders with unified interface."""

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract ReID features from detections.

        Args:
            img (np.ndarray): Full image (H, W, 3) in BGR format.
            dets (np.ndarray): Detections array (N, 5+) [x_center, y_center, w, h, ...].

        Returns:
            (list[np.ndarray]): List of normalized feature vectors.
        """
        raise NotImplementedError


class YOLOReIDEncoder(BaseReIDEncoder):
    """YOLO model as ReID encoder (original implementation)."""

    def __init__(self, model: str):
        """Initialize YOLO ReID encoder.

        Args:
            model (str): Path to YOLO model file (.pt, .engine, .onnx).
        """
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.model(embed=[len(self.model.model.model) - 2 if ".pt" in model else -1], verbose=False, save=False)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract features using YOLO model."""
        feats = self.model.predictor(
            [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        )
        if len(feats) != dets.shape[0] and feats[0].shape[0] == dets.shape[0]:
            feats = feats[0]  # batched prediction with non-PyTorch backend
        return [f.cpu().numpy() for f in feats]


class OSNetONNXReIDEncoder(BaseReIDEncoder):
    """OSNet ReID encoder using ONNX Runtime with TensorRT support.

    This encoder supports multiple formats:
    - Pure ONNX (.onnx) with CUDA/CPU
    - TensorRT via ONNX Runtime TensorRT Execution Provider
    - Pre-built TensorRT engine files (.engine)

    TensorRT acceleration is automatically enabled if available.

    Attributes:
        session: ONNX Runtime inference session.
        input_size (tuple): Model input size (height, width).
        mean (np.ndarray): ImageNet normalization mean.
        std (np.ndarray): ImageNet normalization std.

    Examples:
        Initialize with ONNX model (auto TensorRT)
        >>> encoder = OSNetONNXReIDEncoder("osnet_x1_0.onnx", use_tensorrt=True)

        Initialize with pre-built TensorRT engine
        >>> encoder = OSNetONNXReIDEncoder("osnet_x1_0.engine", use_tensorrt=True)
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (256, 128),
        use_tensorrt: bool = True,
        device: int = 0,
    ):
        """Initialize OSNet ONNX encoder with optional TensorRT acceleration.

        Args:
            model_path (str): Path to .onnx model or .engine file.
            input_size (tuple[int, int]): Input size (height, width).
            use_tensorrt (bool): Enable TensorRT Execution Provider if available.
            device (int): CUDA device ID.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime-gpu is required for OSNet ONNX/TensorRT inference. "
                "Install with: pip install onnxruntime-gpu"
            )

        self.model_path = Path(model_path)
        self.input_size = input_size  # (H, W)
        self.device_id = device

        # Setup execution providers with TensorRT priority
        providers = self._setup_providers(use_tensorrt)

        LOGGER.info(f"Loading OSNet from {self.model_path}")
        LOGGER.info(f"Available providers: {providers}")

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
            actual_provider = self.session.get_providers()[0]
            LOGGER.info(f"✅ OSNet loaded successfully with {actual_provider}")
        except Exception as e:
            LOGGER.warning(f"Failed to load with TensorRT, falling back to CUDA/CPU: {e}")
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            LOGGER.info(f"✅ OSNet loaded with fallback provider: {self.session.get_providers()[0]}")

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Detect input data type (FP32 or FP16)
        input_type = self.session.get_inputs()[0].type
        self.use_fp16 = "float16" in input_type
        if self.use_fp16:
            LOGGER.info(f"✅ Model expects FP16 input, will convert automatically")
        
        # ImageNet normalization parameters
        dtype = np.float16 if self.use_fp16 else np.float32
        self.mean = np.array([0.485, 0.456, 0.406], dtype=dtype).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=dtype).reshape(1, 3, 1, 1)
        
        # Get feature dimension from model output shape
        output_shape = self.session.get_outputs()[0].shape
        self.feat_dim = output_shape[-1] if output_shape[-1] is not None else 512
        
        # Reusable thread pool for preprocessing (lazy initialization)
        self._executor = None

    def _setup_providers(self, use_tensorrt: bool) -> list:
        """Setup ONNX Runtime execution providers with TensorRT priority.

        Args:
            use_tensorrt (bool): Whether to enable TensorRT Execution Provider.

        Returns:
            (list): List of execution providers in priority order.
        """
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]

        providers = []

        # Priority 1: TensorRT Execution Provider (fastest for .engine and .onnx)
        if use_tensorrt and "TensorrtExecutionProvider" in available:
            trt_options = {
                "device_id": self.device_id,
                "trt_max_workspace_size": 4 * (1 << 30),  # 4GB workspace
                "trt_fp16_enable": True,  # Enable FP16 if engine supports it
                "trt_engine_cache_enable": True,  # Cache compiled engines
                "trt_engine_cache_path": str(self.model_path.parent),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(self.model_path.parent),
                # "trt_dla_enable": True,         # 启用DLA
                # "trt_dla_core": 1,
            }
            providers.append(("TensorrtExecutionProvider", trt_options))
            LOGGER.info("🚀 TensorRT Execution Provider enabled")

        # Priority 2: CUDA Execution Provider
        if "CUDAExecutionProvider" in available:
            cuda_options = {
                "device_id": self.device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }
            providers.append(("CUDAExecutionProvider", cuda_options))

        # Priority 3: CPU fallback
        providers.append("CPUExecutionProvider")

        return providers

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> list[np.ndarray]:
        """Extract ReID features from detections using OSNet.

        Args:
            img (np.ndarray): Full image (H, W, 3) in BGR format.
            dets (np.ndarray): Detections array (N, 5+) [x_center, y_center, w, h, ...].

        Returns:
            (list[np.ndarray]): List of L2-normalized feature vectors.
        """
        if len(dets) == 0:
            return []

        # Prepare data
        xyxy_dets = xywh2xyxy(torch.from_numpy(dets[:, :4]))
        num_dets = len(xyxy_dets)
        preprocessed_tensors = [None] * num_dets
        
        # Preprocessing constants (avoid attribute lookup in inner function)
        target_size = (self.input_size[1], self.input_size[0])  # (W, H) for cv2.resize
        mean_val = self.mean
        std_val = self.std
        dtype = np.float16 if self.use_fp16 else np.float32
        img_h, img_w = img.shape[:2]
        input_h, input_w = self.input_size

        def preprocess_one(args):
            i, det = args
            x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return

            # Crop, BGR to RGB, Resize, Normalize in optimized order
            img_crop = img[y1:y2, x1:x2, ::-1]  # Crop + BGR to RGB in one step
            img_resized = cv2.resize(img_crop, target_size)
            
            # Convert to target dtype directly, normalize in one pass
            img_tensor = img_resized.transpose(2, 0, 1).astype(dtype) / 255.0
            img_tensor = img_tensor.reshape(1, 3, input_h, input_w)
            img_tensor = (img_tensor - mean_val) / std_val
            
            preprocessed_tensors[i] = img_tensor

        # Parallel preprocessing for larger batches
        if num_dets > 4:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=4)
            list(self._executor.map(preprocess_one, enumerate(xyxy_dets)))
        else:
            for i, det in enumerate(xyxy_dets):
                preprocess_one((i, det))

        # Sequential inference (Batch Size = 1)
        features = []
        for tensor in preprocessed_tensors:
            if tensor is None:
                # Fallback for empty crop (rare), use zero vector
                features.append(np.zeros(self.feat_dim, dtype=np.float32)) 
                continue

            outputs = self.session.run([self.output_name], {self.input_name: tensor})
            feat = outputs[0].flatten()
            
            # L2 normalization
            feat = feat / (np.linalg.norm(feat) + 1e-12)
            features.append(feat.astype(np.float32))

        return features


def build_reid_encoder(model_type: str, model_path: str, **kwargs):
    """Factory function to build ReID encoder with auto-detection support.

    Args:
        model_type (str): Type of ReID model: "yolo", "osnet-onnx", or "auto".
        model_path (str): Path to model file (.pt, .onnx, .engine).
        **kwargs: Additional arguments:
            - input_size (tuple): Input size (H, W) for OSNet, default (256, 128).
            - use_tensorrt (bool): Enable TensorRT for OSNet, default True.
            - device (int | str): Device ID or "cuda:0"/"cpu", default 0.

    Returns:
        (BaseReIDEncoder): Initialized ReID encoder instance.

    Raises:
        ValueError: If model_type is unknown or file format is unsupported.

    Examples:
        Auto-detect and build YOLO encoder
        >>> encoder = build_reid_encoder("auto", "yolo11n-cls.pt")

        Build OSNet encoder with TensorRT engine
        >>> encoder = build_reid_encoder("osnet-onnx", "osnet_x1_0.engine", use_tensorrt=True)

        Build OSNet encoder with ONNX model
        >>> encoder = build_reid_encoder("osnet-onnx", "osnet_x1_0.onnx", input_size=(256, 128))
    """
    model_path_lower = model_path.lower()

    # Auto-detect model type from file extension and path
    if model_type == "auto":
        if any(x in model_path_lower for x in ["yolo", "cls"]) and model_path_lower.endswith(
            (".pt", ".engine", ".onnx")
        ):
            model_type = "yolo"
        elif model_path_lower.endswith((".onnx", ".engine")):
            model_type = "osnet-onnx"
        else:
            raise ValueError(
                f"Cannot auto-detect model type from path: {model_path}. "
                f"Please specify model_type explicitly: 'yolo' or 'osnet-onnx'"
            )

    # Parse device argument
    device_id = 0
    if "device" in kwargs:
        dev = kwargs["device"]
        if isinstance(dev, int):
            device_id = dev
        elif isinstance(dev, str) and ":" in dev:
            device_id = int(dev.split(":")[1])

    # Build appropriate encoder
    if model_type == "yolo":
        LOGGER.info(f"Building YOLO ReID encoder from {model_path}")
        return YOLOReIDEncoder(model_path)

    elif model_type == "osnet-onnx":
        LOGGER.info(f"Building OSNet ONNX/TensorRT encoder from {model_path}")
        return OSNetONNXReIDEncoder(
            model_path=model_path,
            input_size=kwargs.get("input_size", (256, 128)),
            use_tensorrt=kwargs.get("use_tensorrt", True),
            device=device_id,
        )

    else:
        raise ValueError(
            f"Unknown ReID model type: '{model_type}'. " f"Supported types: 'yolo', 'osnet-onnx', 'auto'"
        )
