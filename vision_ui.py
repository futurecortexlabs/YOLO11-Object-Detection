"""FCTX image AI portfolio, built for Hugging Face Gradio Spaces."""
from pathlib import Path
import gradio as gr
from ultralytics.utils import ASSETS

BASE = Path(__file__).resolve().parent
CONTACT = "https://docs.google.com/forms/d/e/1FAIpQLSeeW9u8w2cvZnnRlUIRfXD-mvKGBvxhFouMYS4FwKCDfhhw4w/viewform?usp=header"

def build(detect_objects, empty_result, columns):
    with gr.Blocks(title="FCTX | 画像AI・物体検出", css=(BASE / "style.css").read_text(encoding="utf-8"),
                   theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate", font=["Arial", "sans-serif"]),
                   analytics_enabled=False, delete_cache=(600, 3600)) as app:
        gr.HTML(f'''<header class="nav"><a class="brand" href="#"><span class="brand-icon">F /</span> FCTX <small>IMAGE INTELLIGENCE</small></a><nav><a href="#experience">画像AIを体験</a><a href="#services">開発できること</a><a class="nav-cta" href="{CONTACT}" target="_blank" rel="noopener noreferrer">開発相談 ↗</a></nav></header>
        <section class="hero"><div class="eyebrow"><span class="dot"></span> FCTX / INTERACTIVE PORTFOLIO</div><h1>画像を、<br>使える<span>情報</span>に。</h1><div class="hero-bottom"><div><p>画像AIのアイデアを、動くWebアプリへ。<br>写真の中の「何が、どこに、いくつ」を可視化。<br>あなたの画像で、その可能性を体験してください。</p><a class="primary-link" href="#experience">画像AIを試す <span>↘</span></a><a class="text-link" href="{CONTACT}" target="_blank" rel="noopener noreferrer">こんなもの作れる？を相談 ↗</a></div><div class="hero-note"><span>POWERED BY</span><strong>YOLO11<span>n</span></strong><p>事前学習済みモデル × Python × Web UI</p></div></div></section>
        <div class="proof-strip"><span>01 <b>画像を選ぶ</b></span><span>02 <b>AIで検出する</b></span><span>03 <b>結果を保存する</b></span></div>
        <section id="experience" class="section-heading"><div class="eyebrow">TRY IT YOURSELF</div><h2>まずは、一枚から。</h2><p>人物・車・動物・日用品など80クラスに対応。サンプル画像でも実際のAI推論を試せます。</p></section>''')
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=280, elem_classes="demo-card"):
                gr.Markdown("### 01 / 画像を選ぶ")
                source = gr.Image(type="pil", sources=["upload"], label="画像をアップロード", height=300)
                gr.Examples(examples=[[str(ASSETS / "bus.jpg")], [str(ASSETS / "zidane.jpg")]], inputs=source, label="画像がない方はこちら / サンプルを選択", cache_examples=False)
                gr.Markdown("JPEG・PNG・WebP / 最大10MB・2,000万画素。サンプル画像：Ultralytics同梱アセット。", elem_classes="muted")
                confidence = gr.Slider(.05, 1, value=.25, step=.05, label="検出の信頼度", info="低くすると候補が増え、高くすると確信度の高い結果に絞ります。")
                with gr.Accordion("詳細設定", open=False):
                    iou = gr.Slider(.1, 1, value=.45, step=.05, label="重複判定（IoU）", info="低くすると重なった検出枠を強く抑制します。")
                with gr.Row():
                    detect = gr.Button("物体検出を実行 →", variant="primary", scale=3)
                    clear = gr.Button("リセット", scale=1)
            with gr.Column(scale=1, min_width=280, elem_classes="demo-card"):
                gr.Markdown("### 02 / 検出結果")
                result = gr.Image(type="pil", label="検出枠つき画像 / 右上から保存", interactive=False, height=380, show_download_button=True)
                with gr.Row():
                    count = gr.Textbox(value="—", label="検出数", interactive=False, min_width=70)
                    kinds = gr.Textbox(value="—", label="種類", interactive=False, min_width=70)
                    timing = gr.Textbox(value="—", label="推論時間", interactive=False, min_width=70)
                status = gr.Textbox(value=empty_result()[2], label="処理状況", interactive=False, lines=3)
                gr.Markdown("初回はモデルの取得・読込に時間がかかります。推論時間にはモデル読込や待機時間を含みません。", elem_classes="muted")
        with gr.Accordion("検出データを見る・CSVで保存", open=False):
            table = gr.Dataframe(value=empty_result()[1], headers=columns, interactive=False, label="クラス名・信頼度（0〜1）・座標（px）")
            csv = gr.File(label="検出結果CSV", interactive=False)
        gr.Markdown("画像はサーバーに送信して処理します。このアプリは画像を学習に使いません。一時ファイルは約1時間後に削除対象となり、10分間隔で削除を確認します。機密情報や個人情報を含む画像はアップロードしないでください。", elem_classes="privacy")
        gr.HTML('''<section id="services"><div class="section-heading"><div class="eyebrow">FROM EXPERIENCE TO YOUR BUSINESS</div><h2>こういう仕組みを、開発できます。</h2><p>AIの検出結果を、業務で使うための画面とデータにつなげます。</p></div><div class="service-grid"><article><span class="card-number">01 / VISION</span><h3>画像から情報を取り出す</h3><p>対象の種類・位置・個数を可視化。撮影条件や目的に合わせ、使えるモデルを検証します。</p><div class="tags">画像認識 / 物体検出 / カウント</div></article><article><span class="card-number">02 / APPLICATION</span><h3>誰でも使えるアプリに</h3><p>アップロードから設定・結果確認・保存まで。使う人の作業に合わせたWeb画面を実装します。</p><div class="tags">Python / Web UI / データ出力</div></article><article><span class="card-number">03 / INTEGRATION</span><h3>業務の流れにつなげる</h3><p>独自対象の学習、カメラ入力、記録・通知・外部システム連携まで、必要な拡張を相談できます。</p><div class="tags">追加学習 / API / システム連携</div></article></div><p class="scope-note">このアプリの実装範囲は、静止画像の物体検出・可視化・CSV出力です。傷や不良品など独自対象の検出、動画・カメラ連携は別途開発・評価が必要です。</p></section>
        <section class="process"><div><div class="eyebrow">HOW WE WORK</div><h2>小さく試して、<br>使える形へ。</h2></div><ol><li><b>01　課題を整理する</b><span>何を見つけたいか、どんな確認作業を楽にしたいか。</span></li><li><b>02　実際の画像で検証する</b><span>検出精度と使い勝手を試作で確かめます。</span></li><li><b>03　業務に合わせて実装する</b><span>画面・データ出力・連携を必要な範囲から整えます。</span></li></ol></section>''')
        with gr.Accordion("よくある質問", open=False):
            gr.Markdown("""**何を検出できますか？**

COCOで学習したYOLO11nを使用します。人物、車、犬、猫、椅子など80クラスが対象です。ヘルメット、傷、不良品などを専用に学習したモデルではありません。

**どのくらい正確ですか？**

明るさ・構図・対象の大きさによって見逃しや誤検出が起きます。信頼度は正解率の保証ではありません。業務導入前に実際のデータで評価します。

**自社向けに変更できますか？**

対象の追加学習、動画対応、カメラ・システム連携など、課題に合わせて検討します。このページはFCTXによるアプリ実装の紹介で、YOLO11モデル自体を独自開発・学習した実績を示すものではありません。

**画像は残りますか？**

入力・結果は処理とダウンロードのため一時保存します。アプリは学習には使用しません。サーバーが稼働中は約1時間を超えたキャッシュを10分間隔で削除します。ホスティング側の管理方針は別途適用されます。
""")
        with gr.Accordion("相談したいことを整理する", open=False):
            gr.Markdown("用途が決まっていなくても大丈夫です。相談メモを作り、フォームへ貼り付けて使えます。")
            purpose = gr.Dropdown(["画像から物体や個数を検出したい", "独自の対象を検出したい", "AIをWebアプリにしたい", "まずはできることを相談したい"], value="まずはできることを相談したい", label="相談したいこと")
            notes = gr.Textbox(label="いま困っていること", placeholder="例：写真に写った商品の数を数えたい", lines=3, max_length=2000)
            make_note = gr.Button("相談メモを作成")
            brief = gr.Textbox(label="コピーして相談フォームへ", interactive=False, show_copy_button=True, lines=6)
            make_note.click(lambda p, n: f"FCTX 画像AI開発の相談\n相談内容：{p}\n課題：{n.strip() or '相談しながら整理したい'}\n\n検出したいもの・画像の撮影条件・希望する使い方をご相談したいです。", [purpose, notes], brief, api_name=False)
            gr.Markdown("メモはフォームへ自動送信されません。")
        gr.HTML(f'''<section class="contact"><div class="eyebrow">LET’S BUILD SOMETHING USEFUL</div><h2>「こんなこと、できる？」<br>から始めましょう。</h2><p>画像AIの活用からWebアプリ開発まで。アイデアの段階からご相談ください。</p><a href="{CONTACT}" target="_blank" rel="noopener noreferrer">開発について相談する ↗</a><small>Googleフォームが開きます</small></section><footer><div><b>FCTX</b><p>AIを、触れられる可能性に。</p></div><div><a href="https://github.com/futurecortexlabs" target="_blank" rel="noopener noreferrer">GitHub ↗</a><a href="https://note.com/future_cortex" target="_blank" rel="noopener noreferrer">note ↗</a><a href="https://huggingface.co/FCTX" target="_blank" rel="noopener noreferrer">Hugging Face ↗</a></div></footer>''')
        outputs = [result, table, status, count, kinds, timing, csv]
        detect.click(detect_objects, [source, confidence, iou], outputs, concurrency_limit=1, concurrency_id="inference", api_name=False)
        for component in [source, confidence, iou]:
            component.change(lambda: empty_result("画像・設定を変更しました。「物体検出を実行」を押してください。"), outputs=outputs, concurrency_id="inference", api_name=False)
        clear.click(lambda: [None, .25, .45, *empty_result()], outputs=[source, confidence, iou, *outputs], concurrency_id="inference", api_name=False).then(lambda: empty_result(), outputs=outputs, concurrency_id="inference", api_name=False)
    return app
