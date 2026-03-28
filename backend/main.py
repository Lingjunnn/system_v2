import asyncio
import base64
import json
import os
import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics.utils.plotting import Annotator

from database import save_fighting_result

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "public", "models")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "data", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


@dataclass
class DetectionResult:
    score: float
    is_fight: bool
    inference_time_ms: float
    timestamp: float


class RingBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque(maxlen=maxlen)

    def append(self, item):
        self.buffer.append(item)

    def clear(self):
        self.buffer.clear()

    def is_full(self) -> bool:
        return len(self.buffer) >= self.buffer.maxlen

    def get_all(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)


class ViolenceDetector:
    GAMMA = 0.67

    def __init__(
        self,
        pose_model_path: str = None,
        violence_model_path: str = None,
        num_frames: int = 20,
        input_size: int = 100,
        threshold: float = 0.7,
    ):
        if pose_model_path is None:
            pose_model_path = os.path.join(MODEL_DIR, "yolo26n-pose.onnx")
        if violence_model_path is None:
            violence_model_path = os.path.join(MODEL_DIR, "violence_detector_standalone.onnx")

        self.num_frames = num_frames
        self.input_size = input_size
        self.threshold = threshold

        self.gamma_table = np.array(
            [((i / 255.0) ** self.GAMMA) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        self.pose_buffer = RingBuffer(maxlen=num_frames)
        self.change_buffer = RingBuffer(maxlen=num_frames)
        self.prev_frame = None
        self.lock = threading.Lock()

        self._init_pose_model(pose_model_path)
        self._init_violence_model(violence_model_path)

    def _init_pose_model(self, model_path: str):
        print(f"[INIT] Loading YOLO26n Pose model from: {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.pose_session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        print("[INIT] YOLO Pose model loaded successfully (CPU)")

    def _init_violence_model(self, model_path: str):
        print(f"[INIT] Loading Violence Detector model from: {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = []
        try:
            available_providers = ort.get_available_providers()
            print(f"[INIT] Available ONNX providers: {available_providers}")

            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("[INIT] Using CUDA GPU acceleration")
            else:
                providers = ['CPUExecutionProvider']
                print("[INIT] CUDA not available, using CPU")
        except Exception as e:
            print(f"[INIT] Error checking providers: {e}")
            providers = ['CPUExecutionProvider']

        self.violence_session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        if 'CUDAExecutionProvider' in self.violence_session.get_providers():
            print("[INIT] Violence Detector model loaded with CUDA GPU acceleration")
        else:
            print("[INIT] Violence Detector model loaded with CPU")

        print(f"[INIT] Model inputs: {self.violence_session.get_inputs()}")
        print(f"[INIT] Model outputs: {self.violence_session.get_outputs()}")

    def extract_pose(self, image: np.ndarray):
        h, w = image.shape[:2]

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)

        outputs = self.pose_session.run(None, {"images": img_batch})

        all_kpts = []

        if len(outputs) > 0:
            output = outputs[0]
            if isinstance(output, list):
                output = np.array(output)
            print(f"[DEBUG] YOLO output[0] shape: {output.shape}")

            if len(output.shape) == 3 and output.shape[1] == 17:
                num_people = output.shape[0]
                print(f"[DEBUG] Detected {num_people} people")
                for person_idx in range(num_people):
                    kpts = np.zeros((17, 3), dtype=np.float32)
                    for i in range(17):
                        kpts[i, 0] = output[person_idx, i, 0] / 640.0 * w
                        kpts[i, 1] = output[person_idx, i, 1] / 640.0 * h
                        kpts[i, 2] = output[person_idx, i, 2]
                    all_kpts.append(kpts)
            elif len(output.shape) == 3 and output.shape[2] == 57:
                print(f"[DEBUG] Parsing YOLO pose output with shape {output.shape}")
                output = output[0]
                for det_idx in range(min(output.shape[0], 10)):
                    det = output[det_idx]
                    kpts = np.zeros((17, 3), dtype=np.float32)
                    for i in range(17):
                        kpts[i, 0] = det[i * 3] / 640.0 * w
                        kpts[i, 1] = det[i * 3 + 1] / 640.0 * h
                        kpts[i, 2] = det[i * 3 + 2]
                    all_kpts.append(kpts)
                print(f"[DEBUG] Found {len(all_kpts)} detections")

        if not all_kpts:
            all_kpts.append(np.zeros((17, 3), dtype=np.float32))

        return all_kpts

    def draw_skeleton(self, all_kpts, orig_w: int, orig_h: int) -> np.ndarray:
        skeleton_img = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        annotator = Annotator(skeleton_img)

        for kpts in all_kpts:
            annotator.kpts(kpts, shape=(orig_h, orig_w))

        return annotator.result()

    def preprocess_pose_frame(self, frame: np.ndarray, all_kpts) -> np.ndarray:
        h, w = frame.shape[:2]
        skeleton_orig = self.draw_skeleton(all_kpts, w, h)

        skeleton_resized = cv2.resize(skeleton_orig, (self.input_size, self.input_size))

        skeleton_rgb = cv2.cvtColor(skeleton_resized, cv2.COLOR_BGR2RGB)
        skeleton_normalized = skeleton_rgb.astype(np.float32) / 255.0

        skeleton_chw = np.transpose(skeleton_normalized, (2, 0, 1))

        return skeleton_chw

    def preprocess_change_frame(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return chw

    def compute_frame_diff(self, current: np.ndarray, previous: Optional[np.ndarray]) -> np.ndarray:
        if previous is None:
            return np.zeros_like(current)

        diff = current - previous
        return diff

    def process_frame(self, frame_data: np.ndarray) -> Optional[DetectionResult]:
        with self.lock:
            print(f"[DEBUG] Input frame: shape={frame_data.shape}, dtype={frame_data.dtype}, range=[{frame_data.min()}, {frame_data.max()}]")

            frame_gamma = cv2.LUT(frame_data, self.gamma_table)
            print(f"[DEBUG] Gamma corrected: shape={frame_gamma.shape}, range=[{frame_gamma.min()}, {frame_gamma.max()}]")

            pose_kpts = self.extract_pose(frame_gamma)
            pose_kpts_array = np.array(pose_kpts) if pose_kpts else np.zeros((1, 17, 3))
            print(f"[DEBUG] Pose keypoints: num_people={len(pose_kpts)}, shape={pose_kpts_array.shape}, sum={pose_kpts_array[:, :, 2].sum():.2f}")

            pose_frame = self.preprocess_pose_frame(frame_gamma, pose_kpts)
            print(f"[DEBUG] Pose frame: shape={pose_frame.shape}, range=[{pose_frame.min():.4f}, {pose_frame.max():.4f}]")

            change_frame_raw = self.preprocess_change_frame(frame_gamma)
            print(f"[DEBUG] Change frame raw: shape={change_frame_raw.shape}, range=[{change_frame_raw.min():.4f}, {change_frame_raw.max():.4f}]")

            if self.prev_frame is not None:
                prev_processed = self.preprocess_change_frame(self.prev_frame)
                change_frame = self.compute_frame_diff(change_frame_raw, prev_processed)
                print(f"[DEBUG] Change frame diff: range=[{change_frame.min():.4f}, {change_frame.max():.4f}]")
            else:
                change_frame = np.zeros_like(change_frame_raw)
                print(f"[DEBUG] First frame, change = zeros")

            self.pose_buffer.append(pose_frame)
            self.change_buffer.append(change_frame)
            self.prev_frame = frame_gamma.copy()

            print(f"[DEBUG] Buffers: pose={len(self.pose_buffer)}/{self.pose_buffer.buffer.maxlen}, change={len(self.change_buffer)}/{self.change_buffer.buffer.maxlen}")

            if not self.pose_buffer.is_full() or not self.change_buffer.is_full():
                print(f"[DEBUG] Buffer not full yet, returning None")
                return None

            return self._run_inference()

    def _run_inference(self) -> Optional[DetectionResult]:
        pose_seq = np.array(self.pose_buffer.get_all(), dtype=np.float32)
        change_seq = np.array(self.change_buffer.get_all(), dtype=np.float32)

        print(f"[DEBUG] pose_seq: shape={pose_seq.shape}, dtype={pose_seq.dtype}, range=[{pose_seq.min():.4f}, {pose_seq.max():.4f}]")
        print(f"[DEBUG] change_seq: shape={change_seq.shape}, dtype={change_seq.dtype}, range=[{change_seq.min():.4f}, {change_seq.max():.4f}]")

        pose_seq = np.expand_dims(pose_seq, axis=0)
        change_seq = np.expand_dims(change_seq, axis=0)

        print(f"[DEBUG] Expanded pose_seq: shape={pose_seq.shape}")
        print(f"[DEBUG] Expanded change_seq: shape={change_seq.shape}")

        start_time = time.perf_counter()

        try:
            outputs = self.violence_session.run(
                None,
                {
                    "pose_seq": pose_seq,
                    "change_seq": change_seq
                }
            )

            inference_time = (time.perf_counter() - start_time) * 1000

            print(f"[DEBUG] Raw outputs: {outputs}")
            if outputs and len(outputs) > 0:
                print(f"[DEBUG] Output[0] shape: {outputs[0].shape}, dtype: {outputs[0].dtype}")
                print(f"[DEBUG] Output[0] content: {outputs[0]}")

                score = float(outputs[0][0])

                provider = self.violence_session.get_providers()[0]
                print(f"[INFERENCE] Score: {score:.4f}, Threshold: {self.threshold}, IsFight: {score > self.threshold}, Time: {inference_time:.1f}ms, Provider: {provider}")

                return DetectionResult(
                    score=score,
                    is_fight=score > self.threshold,
                    inference_time_ms=inference_time,
                    timestamp=time.time()
                )

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()

        return None

    def reset(self):
        with self.lock:
            self.pose_buffer.clear()
            self.change_buffer.clear()
            self.prev_frame = None


app = FastAPI(title="Violence Detection WebSocket Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: Optional[ViolenceDetector] = None
websocket_connection: Optional[WebSocket] = None
inference_thread: Optional[threading.Thread] = None
should_run = threading.Event()


@app.on_event("startup")
async def startup_event():
    global detector
    print("[SERVER] Starting Violence Detection Server...")
    print(f"[SERVER] Model directory: {MODEL_DIR}")
    detector = ViolenceDetector(
        num_frames=20,
        input_size=100,
        threshold=0.7
    )
    print("[SERVER] Server initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    global should_run
    should_run.set()
    print("[SERVER] Shutting down...")


@app.get("/")
async def root():
    return {"message": "Violence Detection WebSocket Server", "status": "running"}


@app.get("/api/records")
async def list_records(camera_id: Optional[str] = None):
    """Get fighting detection records from database."""
    from database import get_all_records, get_records_by_camera

    if camera_id:
        records = get_records_by_camera(camera_id)
    else:
        records = get_all_records()

    return {
        "total": len(records),
        "records": [
            {
                "id": r.id,
                "camera_id": r.camera_id,
                "score": r.score,
                "screenshot_path": r.screenshot_path,
                "created_at": r.created_at.isoformat() if r.created_at else None
            }
            for r in records
        ]
    }


@app.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    global websocket_connection, detector

    await websocket.accept()
    websocket_connection = websocket
    print(f"[WS] Client connected: {websocket.client}")

    current_video_name: Optional[str] = None
    last_frame: Optional[np.ndarray] = None

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "frame":
                    frame_base64 = message.get("data", {}).get("image")
                    video_name = message.get("data", {}).get("videoName")

                    if video_name:
                        current_video_name = video_name

                    if frame_base64:
                        image_data = base64.b64decode(frame_base64)
                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        last_frame = frame.copy()

                        if frame is not None and detector is not None:
                            result = detector.process_frame(frame)

                            if result:
                                screenshot_path = None
                                if result.is_fight and last_frame is not None and current_video_name:
                                    screenshot_path = save_screenshot(
                                        last_frame,
                                        current_video_name,
                                        result.is_fight,
                                        result.score
                                    )

                                    save_fighting_result(
                                        video_name=current_video_name,
                                        score=result.score,
                                        screenshot_path=screenshot_path
                                    )
                                    print(f"[DB] Saved fighting result: {current_video_name} -> score={result.score:.4f}")

                                response = {
                                    "type": "result",
                                    "data": {
                                        "score": float(result.score),
                                        "isFight": result.is_fight,
                                        "inferenceTimeMs": round(result.inference_time_ms, 2),
                                        "timestamp": result.timestamp,
                                        "screenshotPath": screenshot_path
                                    }
                                }
                                await websocket.send_json(response)

                elif message.get("type") == "reset":
                    if detector:
                        detector.reset()
                        current_video_name = None
                        last_frame = None
                        print("[WS] Buffer reset")

                elif message.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except json.JSONDecodeError:
                print("[WS] Invalid JSON received")
            except Exception as e:
                print(f"[WS] Error processing frame: {e}")

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {websocket.client}")
    except Exception as e:
        print(f"[WS] WebSocket error: {e}")
    finally:
        websocket_connection = None


def save_screenshot(
    frame: np.ndarray,
    video_name: str,
    is_fight: bool,
    score: float
) -> str:
    """
    保存截图到本地

    Args:
        frame: 原始帧
        video_name: 视频名称
        is_fight: 是否为打斗
        score: 置信度分数

    Returns:
        str: 截图保存路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "fight" if is_fight else "normal"
    filename = f"{video_name}_{label}_{score:.2f}_{timestamp}.jpg"
    filepath = os.path.join(SCREENSHOT_DIR, filename)

    cv2.imwrite(filepath, frame)
    return filepath


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
