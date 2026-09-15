"""PPE inference, conservative person association, and session-scoped reports."""
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
import html
import logging
import os
import tempfile

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
REPO = "melihuzunoglu/ppe-detection"
REVISION = "aed7807c358836eef5a6f67af27dd970ade6fb18"
LOCK = Lock()
HEADERS = ["対象", "ヘルメット", "安全ベスト", "確認結果", "備考"]
RAW_HEADERS = ["class_name", "confidence", "x_min", "y_min", "x_max", "y_max"]
INITIAL = "サンプル体験、または現場画像のAI解析を選んでください。"

@lru_cache(maxsize=1)
def model():
    custom = os.environ.get("PPE_MODEL_PATH")
    path = custom or hf_hub_download(REPO, "best.pt", revision=REVISION, local_dir=BASE / "models" / "ppe")
    loaded = YOLO(str(path))
    names = set(loaded.names.values())
    if not {"human", "helmet", "no-helmet", "vest"}.issubset(names):
        raise ValueError("Model must expose human, helmet, no-helmet and vest classes")
    return loaded

def assess(detections, required):
    """Assign each PPE box to at most one person; unmatched/ambiguous PPE is unknown."""
    people = [d for d in detections if d[0] == "human"]
    assigned = [[] for _ in people]
    for gear in [d for d in detections if d[0] != "human"]:
        name, _, x1, y1, x2, y2 = gear
        cx, cy = (x1+x2)/2, (y1+y2)/2
        candidates = []
        for i, p in enumerate(people):
            _, _, px1, py1, px2, py2 = p
            relative_y = (cy-py1) / max(py2-py1, 1)
            zone = (0 <= relative_y <= .45) if name in ("helmet", "no-helmet") else (.15 <= relative_y <= .8)
            if px1 <= cx <= px2 and py1 <= cy <= py2 and zone:
                candidates.append(i)
        if len(candidates) == 1:
            assigned[candidates[0]].append(name)
    rows = []
    for i, labels in enumerate(assigned):
        helmet = "競合・要確認" if "helmet" in labels and "no-helmet" in labels else "未着用候補" if "no-helmet" in labels else "検出" if "helmet" in labels else "未確認"
        vest = "検出" if "vest" in labels else "未確認"
        states = [helmet if r == "ヘルメット" else vest for r in required]
        result = "対象外" if not states else "要確認" if any(s != "検出" for s in states) else "装備検出"
        note = "未確認は未着用の断定ではありません" if result == "要確認" else "指定装備の検出結果。安全の保証ではありません"
        rows.append([f"作業者 {i+1:02}", helmet, vest, result, note])
    return rows

def sample_scene():
    """Original schematic, with deterministic demonstration annotations (no inference)."""
    image = Image.new("RGB", (1200, 700), "#192c37")
    d = ImageDraw.Draw(image)
    for x in range(0, 1200, 100):
        d.line((x, 0, x, 700), fill="#233a45", width=1)
    for y in range(0, 700, 100):
        d.line((0, y, 1200, y), fill="#233a45", width=1)
    d.rectangle((0, 460, 1200, 700), fill="#263e46")
    for x in (50, 960):
        d.rectangle((x, 140, x+190, 430), fill="#344d55", outline="#49646c", width=3)
        for y in (210, 320):
            d.line((x, y, x+190, y), fill="#60747a", width=5)
    rows = []
    for i, x in enumerate((340, 570, 800)):
        d.ellipse((x-26, 170, x+26, 232), fill="#d8ae8c")
        d.rounded_rectangle((x-45, 237, x+45, 398), 16, fill="#879ba8")
        d.line((x-25, 390, x-31, 536), fill="#111f2c", width=30)
        d.line((x+25, 390, x+31, 536), fill="#111f2c", width=30)
        d.line((x-45, 255, x-64, 369), fill="#879ba8", width=23)
        d.line((x+45, 255, x+64, 369), fill="#879ba8", width=23)
        rows.append(["human", .96, x-75, 155, x+75, 550])
        if i != 1:
            d.pieslice((x-35, 153, x+35, 213), 180, 360, fill="#efd16e")
            d.rectangle((x-39, 178, x+39, 188), fill="#efd16e")
            rows.append(["helmet", .94, x-39, 153, x+39, 188])
        else:
            rows.append(["no-helmet", .89, x-28, 163, x+28, 225])
        if i != 2:
            d.polygon([(x-42,238),(x-15,238),(x,270),(x+15,238),(x+42,238),(x+40,391),(x-40,391)], fill="#dfaa43")
            d.rectangle((x-39,320,x+39,331), fill="#fff4c5")
            d.line((x-23,245,x-23,390), fill="#fff4c5", width=7)
            d.line((x+23,245,x+23,390), fill="#fff4c5", width=7)
            rows.append(["vest", .92, x-42,238,x+42,391])
    return image, rows

def annotate(image, detections, rows):
    image = image.copy()
    d = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(14, image.width//65))
    for index, person in enumerate(p for p in detections if p[0] == "human"):
        color = "#fbbf60" if rows[index][3] == "要確認" else "#71e5c0"
        coords = tuple(person[2:])
        d.rectangle(coords, outline=color, width=max(2,image.width//300))
        label = f"PERSON {index+1:02} / " + ("REVIEW" if rows[index][3] == "要確認" else "DETECTED" if rows[index][3] == "装備検出" else "NOT EVALUATED")
        x, y = coords[0], max(0, coords[1]-28)
        bounds = d.textbbox((x,y),label,font=font)
        d.rectangle((bounds[0]-2,bounds[1]-2,bounds[2]+5,bounds[3]+3),fill=color)
        d.text((x,y),label,fill="#142b32",font=font)
    return image

def metrics(total="—", review="—", found="—", elapsed="—"):
    return '<div class="metrics">' + ''.join(f'<div><span>{label}</span><strong>{value}</strong><small>{unit}</small></div>' for label,value,unit in [("検出した作業者",total,"人"),("確認が必要",review,"人"),("指定装備を検出",found,"人"),("解析時間",elapsed,"秒")]) + '</div>'

def blank(message=INITIAL):
    return None, metrics(), pd.DataFrame(columns=HEADERS), pd.DataFrame(columns=RAW_HEADERS), message, None, None

def run(image, confidence, required, sample, cache_dir):
    if not sample and image is None:
        return blank("画像が未入力です。現場画像をアップロードしてください。")
    try:
        if not .05 <= float(confidence) <= 1:
            return blank("信頼度を0.05〜1.00に設定してください。")
        if not required or any(r not in ["ヘルメット", "安全ベスト"] for r in required):
            return blank("確認する装備を1つ以上選択してください。")
        started = perf_counter()
        if sample:
            image, detections = sample_scene()
            mode = "シミュレーション（固定の説明用データ・AI推論なし）"
        else:
            if image.width * image.height > 20_000_000:
                return blank("画像は2,000万画素以下に縮小してください。")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((1920,1920))
            with LOCK:
                result = model().predict(image, conf=float(confidence), iou=.45, verbose=False, save=False)[0]
                detections = [[result.names[int(b.cls[0].item())], round(float(b.conf[0].item()),4), *[round(float(v),2) for v in b.xyxy[0].tolist()]] for b in result.boxes] if result.boxes is not None else []
            mode = "AI解析 / YOLO11 PPE"
        detections = [d for d in detections if d[1] >= float(confidence)]
        rows = assess(detections, required)
        review = sum(r[3] == "要確認" for r in rows)
        found = sum(r[3] == "装備検出" for r in rows)
        elapsed = round(perf_counter()-started,2)
        annotated = annotate(image, detections, rows)
        table = pd.DataFrame(rows, columns=HEADERS)
        raw = pd.DataFrame(detections, columns=RAW_HEADERS)
        message = f"{mode}｜作業者 {len(rows)}人・要確認 {review}人。"
        if not rows:
            message += " 作業者が検出されず、装備の確認はできませんでした。"
        if any(d[0] == "no-helmet" for d in detections) and not any(r[1] == "未着用候補" for r in rows):
            message += " 人物に関連付けできない、または競合する未着用候補があります。生データを確認してください。"
        message += " 未確認には遮蔽・画角・見逃しを含みます。"
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        report_table = table.copy()
        report_table.insert(0,"解析モード",mode)
        report_table.insert(1,"確認対象", " / ".join(required))
        with tempfile.NamedTemporaryFile(mode="w",suffix=".csv",prefix="ppe-",dir=cache,delete=False,encoding="utf-8-sig",newline="") as f:
            report_table.to_csv(f,index=False)
            csv_path = f.name
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        report = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>PPE確認レポート</title><style>body{{font:15px sans-serif;max-width:1000px;margin:50px auto;color:#18382f}}table{{border-collapse:collapse;width:100%;margin:24px 0}}td,th{{padding:12px;border:1px solid #ddd;text-align:left}}small{{color:#666}}</style><h1>SAFE SIGHT / PPE確認レポート</h1><p>FCTX · {timestamp}</p><p>{html.escape(message)}</p><p>確認対象: {html.escape(' / '.join(required))} / 信頼度しきい値: {confidence}</p>{table.to_html(index=False,escape=True)}<h2>モデル検出データ</h2>{raw.to_html(index=False,escape=True)}<p>座標は出力画像 {image.width} × {image.height}px 基準です。未検出は未着用を意味しません。現場担当者による確認を支援する参考情報です。</p><small>Model: {REPO} @ {REVISION if not os.environ.get('PPE_MODEL_PATH') else 'custom'} · 出典と利用条件はREADME参照</small></html>'''
        with tempfile.NamedTemporaryFile(mode="w",suffix=".html",prefix="ppe-report-",dir=cache,delete=False,encoding="utf-8") as f:
            f.write(report)
            report_path = f.name
        return annotated, metrics(len(rows),review,found,"—" if sample else f"{elapsed:.2f}"), table, raw, message, csv_path, report_path
    except Exception:
        logging.exception("PPE analysis failed")
        return blank("解析に失敗しました。画像を変更して再試行してください。初回は専用モデルの取得にネット接続が必要です。サンプル体験はオフラインでも利用できます。")
