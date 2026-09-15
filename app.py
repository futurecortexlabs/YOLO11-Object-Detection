"""FCTX: image AI portfolio powered by YOLO11."""
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
import logging
import os
import tempfile

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "fctx-ultralytics"))
import gradio as gr
import pandas as pd
from PIL import Image, ImageOps
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "yolo11n.pt"
CONTACT = "https://docs.google.com/forms/d/e/1FAIpQLSeeW9u8w2cvZnnRlUIRfXD-mvKGBvxhFouMYS4FwKCDfhhw4w/viewform?usp=header"
COLUMNS = ["class_name", "confidence", "x_min", "y_min", "x_max", "y_max"]
INFERENCE_LOCK = Lock()
INITIAL = "画像を選んで「物体検出を実行」を押してください。"

@lru_cache(maxsize=1)
def load_model():
    return YOLO(str(MODEL_PATH))

def create_empty_table():
    return pd.DataFrame(columns=COLUMNS)

def empty_result(message=INITIAL):
    return None, create_empty_table(), message, "—", "—", "—", None

def detect_objects(input_image, confidence_threshold, iou_threshold):
    if input_image is None:
        return empty_result("画像が未入力です。画像をアップロードしてください。")
    try:
        if not 0.05 <= float(confidence_threshold) <= 1 or not 0.1 <= float(iou_threshold) <= 1:
            return empty_result("しきい値を設定範囲内にしてください。")
        image = input_image if isinstance(input_image, Image.Image) else Image.fromarray(input_image)
        if image.width * image.height > 20_000_000:
            return empty_result("画像が大きすぎます。2,000万画素以下に縮小してください。")
        image = ImageOps.exif_transpose(image).convert("RGB").copy()
        # Bound inference and drawing memory; exported coordinates match this resized image.
        original_size = image.size
        image.thumbnail((1920, 1920))
        with INFERENCE_LOCK:
            model = load_model()
            started = perf_counter()
            result = model.predict(source=image, conf=float(confidence_threshold),
                                   iou=float(iou_threshold), verbose=False, save=False)[0]
            elapsed = perf_counter() - started
            annotated = Image.fromarray(result.plot()[..., ::-1])
            rows = []
            if result.boxes is not None:
                for box in result.boxes:
                    rows.append([result.names[int(box.cls[0].item())],
                                 round(float(box.conf[0].item()), 4),
                                 *[round(float(v), 2) for v in box.xyxy[0].tolist()]])
        table = pd.DataFrame(rows, columns=COLUMNS).sort_values("confidence", ascending=False)
        message = (f"検出完了：{len(rows)}個の物体を検出しました。" if rows else
                   "検出完了：物体は見つかりませんでした。信頼度を下げるか、別の画像をお試しください。")
        if image.size != original_size:
            message += f" 画像を{image.width}×{image.height}pxに縮小しました。座標は縮小後の画像基準です。"
        # Write exports directly into Gradio's managed cache, which expires after an hour.
        Path(demo.GRADIO_CACHE).mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix="detections-",
                                         dir=demo.GRADIO_CACHE, delete=False, encoding="utf-8-sig", newline="") as export:
            table.to_csv(export, index=False)
            csv_path = export.name
        return annotated, table, message, str(len(rows)), str(table.class_name.nunique()), f"{elapsed:.2f} 秒", csv_path
    except Exception:
        logging.exception("Object detection failed")
        return empty_result("処理に失敗しました。画像を変更して再試行してください。初回はモデルの取得にインターネット接続が必要です。")

from vision_ui import build

demo = build(detect_objects, empty_result, COLUMNS)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), max_file_size="10mb", show_error=False)
