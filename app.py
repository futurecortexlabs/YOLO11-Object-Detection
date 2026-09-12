"""Future Cortex Labs: image AI portfolio and working YOLO11 demo."""
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
CONTACT = "https://forms.gle/SaWGZFu8J7DgbytL7"
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
        image = ImageOps.exif_transpose(image).convert("RGB")
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

CSS = (BASE / "style.css").read_text(encoding="utf-8")
with gr.Blocks(title="画像AI開発・物体検出デモ | Future Cortex Labs", css=CSS,
               theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", font=["Arial", "sans-serif"]),
               analytics_enabled=False, delete_cache=(600, 3600)) as demo:
    gr.HTML(f'''<header class="nav"><a class="brand" href="#"><span class="brand-icon">F /</span> FUTURE CORTEX <small>LABS</small></a><nav><a href="#experience">デモを体験</a><a href="#services">できること</a><a class="nav-cta" href="{CONTACT}" target="_blank" rel="noopener noreferrer">開発の相談 ↗</a></nav></header>
    <section class="hero"><div class="eyebrow"><span class="dot"></span> IMAGE AI / INTERACTIVE PORTFOLIO</div><h1>画像を、<br>使える<span>情報</span>に。</h1><div class="hero-bottom"><div><p>画像AIのアイデアを、動くWebアプリへ。<br>物体検出から結果の可視化まで。まずは、あなたの画像で体験してください。</p><a class="primary-link" href="#experience">物体検出を試す <span>↘</span></a><a class="text-link" href="{CONTACT}" target="_blank" rel="noopener noreferrer">こんなもの作れる？を相談 ↗</a></div><div class="hero-note"><span>POWERED BY</span><strong>YOLO11<span>n</span></strong><p>事前学習済みモデル × Python × Web UI</p></div></div></section>
    <div class="proof-strip"><span>01 <b>画像をアップロード</b></span><span>02 <b>AIが物体を検出</b></span><span>03 <b>結果を確認・保存</b></span></div>
    <section id="experience" class="section-heading"><div class="eyebrow">TRY IT YOURSELF</div><h2>画像AIを、ここで体験。</h2><p>人物・車・動物・日用品など、COCOの80クラスを検出します。</p></section>''')
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=300, elem_classes="demo-card"):
            gr.Markdown("### 01 / 画像を選ぶ")
            input_image = gr.Image(label="画像をアップロード", type="pil", sources=["upload"], height=330)
            gr.Markdown("JPEG・PNG・WebP / 最大10MB・2,000万画素。人物や車などがはっきり写った写真がおすすめです。", elem_classes="muted")
            confidence = gr.Slider(0.05, 1, value=0.25, step=0.05, label="検出の信頼度", info="低くすると多く検出、高くすると確信度の高い結果に絞ります。")
            with gr.Accordion("詳細設定", open=False):
                iou = gr.Slider(0.1, 1, value=0.45, step=0.05, label="重複判定（IoU）", info="低くすると重なった検出枠をより強く抑制します。")
            with gr.Row():
                detect = gr.Button("物体検出を実行 →", variant="primary", scale=3)
                clear = gr.Button("リセット", scale=1)
        with gr.Column(scale=1, min_width=300, elem_classes="demo-card"):
            gr.Markdown("### 02 / 検出結果")
            output = gr.Image(label="検出枠つき画像 / 右上から保存", type="pil", interactive=False, height=330, show_download_button=True)
            with gr.Row():
                count = gr.Textbox(value="—", label="検出数", interactive=False, min_width=80)
                classes = gr.Textbox(value="—", label="種類", interactive=False, min_width=80)
                timing = gr.Textbox(value="—", label="処理時間", interactive=False, min_width=80)
            status = gr.Textbox(value=INITIAL, label="処理状況", interactive=False, lines=2)
            gr.Markdown("初回はモデルを取得するため時間がかかります。処理時間はモデル読込後の推論時間です。", elem_classes="muted")
    with gr.Accordion("検出データを見る・CSVで保存", open=False):
        table = gr.Dataframe(headers=COLUMNS, value=create_empty_table(), interactive=False, label="クラス名・信頼度（0〜1）・検出枠の座標（px）")
        download = gr.File(label="検出結果CSV", interactive=False)
    gr.Markdown("画像は推論のためサーバーへ送信され、一時ファイルは約1時間後（削除確認は10分間隔）に削除されます。このアプリは画像を学習に使用しません。公開デモには機密情報・個人情報を含む画像を送信しないでください。", elem_classes="privacy")
    gr.HTML(f'''<section id="services" class="services"><div class="section-heading"><div class="eyebrow">FROM DEMO TO YOUR BUSINESS</div><h2>この体験を、あなたの業務へ。</h2><p>画像の入力から、AI処理、使いやすい画面まで。一連の仕組みを開発します。</p></div><div class="service-grid"><article><span class="card-number">01 / VISION</span><h3>画像認識を組み込む</h3><p>写真から対象を検出し、種類や個数を可視化。確認作業を支援する画像AIツールへ。</p><div class="tags">物体検出 / カウント / 可視化</div></article><article><span class="card-number">02 / APPLICATION</span><h3>誰でも使える画面に</h3><p>AIモデルを動かすだけでなく、アップロード・設定・結果保存まで扱えるWebアプリへ。</p><div class="tags">Python / Web UI / CSV出力</div></article><article><span class="card-number">03 / CUSTOMIZE</span><h3>業務に合わせて育てる</h3><p>独自の検出対象、カメラ連携、既存システム連携など、目的に合わせた拡張を相談できます。</p><div class="tags">要件整理 / PoC / システム連携</div></article></div><p class="scope-note">このデモの実装範囲は、静止画像の物体検出と結果出力です。傷・不良品など独自対象の検出には、別途データ収集・学習・評価が必要です。</p></section>
    <section class="process"><div><div class="eyebrow">HOW WE WORK</div><h2>小さく試して、<br>使える形へ。</h2></div><ol><li><b>01　課題を聞く</b><span>何を見つけたいか、どの作業を楽にしたいかを整理。</span></li><li><b>02　動く試作で確かめる</b><span>実際の画像で、精度と使い勝手を検証。</span></li><li><b>03　業務に合う形にする</b><span>必要な機能と運用方法を決めて実装。</span></li></ol></section>''')
    with gr.Accordion("よくある質問", open=False):
        gr.Markdown("""**何を検出できますか？**
人物、自動車、犬、猫、椅子など、COCOの80クラスです。任意の物体や傷を自動的に検出するものではありません。

**検出結果は正確ですか？**
画像の明るさ・構図・対象の大きさにより見逃しや誤検出があります。信頼度は正解率の保証ではありません。業務導入前に実データで評価します。

**自社向けに変更できますか？**
対象の追加、データ出力や外部連携など、必要な機能を相談フォームでお知らせください。実現方法を検討します。

**相談時に何を伝えればよいですか？**
「検出したいもの」「今の作業」「実現したいこと」の3点だけでも大丈夫です。""")
    gr.HTML(f'''<section class="contact"><div class="eyebrow">LET’S BUILD SOMETHING USEFUL</div><h2>「こんなこと、できる？」<br>から始めましょう。</h2><p>画像AIの活用やWebアプリ開発。アイデアの段階からご相談ください。</p><a href="{CONTACT}" target="_blank" rel="noopener noreferrer">開発について相談する ↗</a><small>Googleフォームが開きます</small></section><footer><div><b>FUTURE CORTEX LABS</b><p>AIを、触れられる可能性に。</p></div><div><a href="https://github.com/futurecortexlabs" target="_blank" rel="noopener noreferrer">GitHub ↗</a><a href="https://note.com/future_cortex" target="_blank" rel="noopener noreferrer">note ↗</a><a href="https://huggingface.co/FCTX" target="_blank" rel="noopener noreferrer">Hugging Face ↗</a></div></footer>''')
    outputs = [output, table, status, count, classes, timing, download]
    detect.click(detect_objects, [input_image, confidence, iou], outputs, concurrency_limit=1, concurrency_id="inference", api_name=False)
    clear.click(lambda: [None, 0.25, 0.45, *empty_result()], outputs=[input_image, confidence, iou, *outputs], concurrency_id="inference", api_name=False)
    input_image.change(lambda: empty_result(), outputs=outputs, concurrency_id="inference", api_name=False)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), max_file_size="10mb", show_error=False)
