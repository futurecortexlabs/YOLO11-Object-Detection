"""Japanese interactive portfolio for FCTX."""
from pathlib import Path
import os
import tempfile
import gradio as gr
import ppe

BASE = Path(__file__).resolve().parent
CONTACT = "https://docs.google.com/forms/d/e/1FAIpQLSeeW9u8w2cvZnnRlUIRfXD-mvKGBvxhFouMYS4FwKCDfhhw4w/viewform?usp=header"

def build():
    with gr.Blocks(title="SAFE SIGHT | PPE安全監視AI · FCTX", css=(BASE / "ppe.css").read_text(encoding="utf-8"), theme=gr.themes.Soft(primary_hue="emerald",neutral_hue="slate",font=["Arial","sans-serif"]), analytics_enabled=False, delete_cache=(600,3600)) as demo:
        gr.HTML(f'''<header class="nav"><a class="brand" href="#"><span class="brand-mark">S<span>◉</span></span><span>SAFE SIGHT<small>by FCTX</small></span></a><nav><a href="#workspace">画像AIを体験</a><a href="#solutions">開発できること</a><a class="nav-contact" href="{CONTACT}" target="_blank" rel="noopener noreferrer">導入・開発の相談 ↗</a></nav></header>
        <section class="hero"><div class="hero-copy"><div class="eyebrow"><span class="dot"></span> COMPUTER VISION FOR WORKPLACE SAFETY</div><h1>現場の安全を、<br><em>見える</em>仕組みに。</h1><p>ヘルメット・安全ベストの着用確認をAIでサポート。<br>画像の解析から、確認箇所の可視化、レポートまで。<br>現場で使うためのAIアプリを、ここで体験できます。</p><div class="hero-actions"><a class="primary-link" href="#workspace">画像AIを体験 <span>↘</span></a><a class="text-link" href="#solutions">このようなシステムを開発できます →</a></div><div class="hero-meta"><span>01　PPE検出</span><span>02　確認箇所の可視化</span><span>03　レポート出力</span></div></div>
        <div class="hero-panel"><div class="panel-top"><span>SAFETY INTELLIGENCE</span><span class="chip">INTERACTIVE EXPERIENCE</span></div><div class="radar"><div class="radar-ring"></div><div class="scan-corner tl"></div><div class="scan-corner br"></div><svg viewBox="0 0 240 220" aria-label="ヘルメットと安全ベストのイラスト" role="img"><path d="M77 80a43 43 0 0 1 86 0" fill="#e8ce88"/><path d="M72 80h96M120 38v36" stroke="#fcf1cc" stroke-width="9" stroke-linecap="round"/><path d="M92 100l28 23 28-23 28 19 12 78H52l12-78z" fill="#8eb4a2"/><path d="M90 105v91m60-91v91M60 167h120" stroke="#e0efc2" stroke-width="9"/><path d="M120 124v73" stroke="#23483c" stroke-width="4"/></svg><span class="gear-tag tag-a">✓ HELMET</span><span class="gear-tag tag-b">✓ SAFETY VEST</span></div><div class="panel-bottom"><span>見つける。確かめる。改善する。</span><small>画像AI × 業務アプリケーション</small></div></div></section>
        <div class="industry-strip"><span>DESIGNED FOR</span><b>製造・工場</b><i>/</i><b>建設・施工</b><i>/</i><b>物流・倉庫</b><span class="strip-end">現場に合わせたカスタム開発 ↗</span></div>
        <section id="workspace" class="section-heading"><div><div class="eyebrow">01 / INTERACTIVE WORKSPACE</div><h2>安全監視AIを、体験する。</h2><p>まずはサンプルで流れを確認。お手元の画像では、PPE専用モデルが解析します。</p></div><span class="workspace-badge">●　ブラウザで体験</span></section>''')
        with gr.Tabs():
            with gr.Tab("PPEモニター", id="monitor"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=7, min_width=300, elem_classes="monitor-card"):
                        gr.HTML('<div class="monitor-heading"><b>VISUAL MONITOR</b><span>STATIC IMAGE ANALYSIS</span></div>')
                        initial_image, initial_detections = ppe.sample_scene()
                        initial_image = ppe.annotate(initial_image,initial_detections,ppe.assess(initial_detections,["ヘルメット","安全ベスト"]))
                        output = gr.Image(value=initial_image, label="解析ビュー / 初期表示は説明用イラスト", type="pil", interactive=False, height=390, show_download_button=True)
                        gr.HTML('<div class="legend"><span><i class="green"></i>指定装備を検出</span><span><i class="amber"></i>要確認</span><small>初期画像・サンプルはシミュレーション</small></div>')
                    with gr.Column(scale=3, min_width=280, elem_classes="control-card"):
                        gr.Markdown("### 解析コントロール")
                        sample = gr.Button("サンプルで体験 →", variant="primary")
                        gr.Markdown("画像なしですぐ体験。固定データを使い、確認・保存の流れを再現します。", elem_classes="muted")
                        with gr.Accordion("現場画像でAI解析",open=True):
                            input_image = gr.Image(label="現場画像 / カメラで静止画撮影",type="pil",sources=["upload","webcam"],height=170)
                            detect = gr.Button("この画像をAI解析",variant="primary")
                        required = gr.CheckboxGroup(["ヘルメット","安全ベスト"],value=["ヘルメット","安全ベスト"],label="確認する装備")
                        with gr.Accordion("検出設定",open=False):
                            confidence = gr.Slider(.05,1,value=.25,step=.05,label="信頼度しきい値",info="低くすると候補が増えます。値は正解率ではありません。")
                        clear = gr.Button("結果をリセット",size="sm")
                kpis = gr.HTML(ppe.metrics())
                status = gr.Textbox(value=ppe.INITIAL,label="解析ステータス",interactive=False,lines=2)
                gr.Markdown("**見方**：未確認は、未着用・遮蔽・画角外・見逃しを区別できない状態です。装備検出も安全を保証するものではありません。人物への関連付けが曖昧な検出は集計に含めず、生データに残します。",elem_classes="method-note")
                gr.Markdown("### 作業者ごとの確認結果")
                table = gr.Dataframe(value=ppe.blank()[2],headers=ppe.HEADERS,interactive=False,label="指定した装備を基準に集計 / IDはこの画像内のみ")
                with gr.Row():
                    csv = gr.File(label="確認結果をCSVで保存",interactive=False)
                    report = gr.File(label="レポートを保存（HTML・ブラウザから印刷可）",interactive=False)
                with gr.Accordion("モデルの検出生データ",open=False):
                    raw = gr.Dataframe(value=ppe.blank()[3],headers=ppe.RAW_HEADERS,interactive=False,label="信頼度 / 出力画像基準の座標（px）")
                gr.Markdown("画像はサーバーで解析します。最大10MB・2,000万画素、長辺1920pxまで縮小。一時ファイルは1時間経過後、10分間隔で削除確認します。このアプリは画像を学習に使用しません。",elem_classes="muted")
            with gr.Tab("導入プランを整理"):
                gr.Markdown("### あなたの現場なら、何から始める？\n選んだ内容から相談用のメモを作成できます。外部には送信されません。")
                with gr.Row():
                    industry = gr.Dropdown(["製造・工場","建設・施工","物流・倉庫","その他"],value="製造・工場",label="現場の種類")
                    source = gr.Dropdown(["静止画像で検証したい","録画した動画を解析したい","既存カメラに接続したい"],value="静止画像で検証したい",label="希望する使い方")
                needs = gr.CheckboxGroup(["ヘルメット","安全ベスト","保護メガネ","手袋","安全靴"],value=["ヘルメット","安全ベスト"],label="検討したい装備（追加装備は個別開発）")
                notes = gr.Textbox(label="解決したい課題",placeholder="例：工場入口での装備確認を効率化したい",lines=3,max_length=2000)
                brief_button = gr.Button("相談メモを作成",variant="primary")
                brief = gr.Textbox(label="相談フォームに貼り付けて使えます",interactive=False,lines=10,show_copy_button=True)
                brief_file = gr.File(label="相談メモをダウンロード",interactive=False)
                gr.HTML(f'<p class="brief-contact"><a href="{CONTACT}" target="_blank" rel="noopener noreferrer">メモをもとに開発相談へ ↗</a><small>Googleフォームが開きます。メモは自動送信されません。</small></p>')
            with gr.Tab("このアプリの実装範囲"):
                gr.Markdown("""### 動く機能と、現場導入で育てる機能

| このアプリで体験できること | 個別開発で対応すること |
| --- | --- |
| PPE専用YOLO11による静止画像解析 | 録画動画・RTSPカメラの継続監視 |
| 人物・ヘルメット・未着用候補・安全ベストの検出 | メガネ・手袋・安全靴の追加学習 |
| 人物と装備の関連付け、要確認の可視化 | 現場別ルール、エリア・時刻条件 |
| カメラから静止画撮影、画像アップロード | メール・チャット通知、外部API連携 |
| CSV・印刷できるHTMLレポート | 履歴DB、権限管理、複数拠点ダッシュボード |

### 判定の仕組み
装備の中心点が人物枠内の適切な領域にあり、候補人物が1人の場合に関連付けます。複数人物に重なる場合や、着用と未着用候補が競合する場合は確認を促します。ベスト未検出を未着用と断定しません。

### 使用モデル
[melihuzunoglu/ppe-detection](https://huggingface.co/melihuzunoglu/ppe-detection) の公開学習済みYOLO11モデルを使用しています。FCTXによる独自学習の実績や、実証済み精度を示すものではありません。本アプリではアプリ設計・推論連携・確認フロー・帳票化を実装しています。モデル取得元・固定バージョン・利用条件はREADMEに記載しています。

### 実運用に向けて
導入する現場の画像で検出漏れ・誤検出・装備の関連付けを評価し、担当者の確認フローと一緒に設計します。このアプリは常時監視・緊急通報を行いません。""")
        gr.HTML(f'''<section id="solutions" class="solutions"><div class="section-heading"><div><div class="eyebrow">02 / WHAT I CAN BUILD</div><h2>AIモデルを、業務で使えるソフトへ。</h2><p>「検出できた」で終わらせず、現場で確認し、次の行動につながる仕組みをつくります。</p></div></div><div class="solution-grid"><article><span class="solution-icon">⌖</span><small>VISION / 01</small><h3>現場に合わせた画像AI</h3><p>検出したい対象や撮影条件を整理。公開モデルの検証から、独自対象の学習まで設計します。</p><div>物体検出 / PPE / カウント</div></article><article><span class="solution-icon">▤</span><small>APPLICATION / 02</small><h3>判断しやすい操作画面</h3><p>確認箇所がひと目でわかるモニター、設定、データ出力。使う人の作業に合わせて実装します。</p><div>Webアプリ / 可視化 / 帳票</div></article><article><span class="solution-icon">⇄</span><small>INTEGRATION / 03</small><h3>運用につながる連携</h3><p>カメラ接続、通知、記録の蓄積まで。導入環境と要件に合わせて段階的に拡張できます。</p><div>カメラ / API / 運用設計</div></article></div></section>
        <section class="process"><div><div class="eyebrow">03 / BUILD WITH YOU</div><h2>まずは、ひとつの現場から。</h2><p>画像と課題があれば、検証を始められます。</p></div><ol><li><b><span>01</span> 課題・撮影環境を整理</b><p>何を確認したいか、誰が使うかを一緒に整理。</p></li><li><b><span>02</span> 実画像で小さく検証</b><p>見逃しや誤検出を確認し、使える条件を見極めます。</p></li><li><b><span>03</span> 業務の流れに合わせて開発</b><p>画面・連携・運用まで、必要な範囲を実装します。</p></li></ol></section>
        <section class="contact"><div><div class="eyebrow">LET’S MAKE IT WORK.</div><h2>「うちの現場でも使える？」<br>その相談から、始めましょう。</h2><p>PPE検出に限らず、画像AI・業務アプリの開発をご相談いただけます。</p></div><div><a href="{CONTACT}" target="_blank" rel="noopener noreferrer">開発について相談する ↗</a><small>要件がまとまっていなくても大丈夫です。<br>Googleフォームが開きます。</small></div></section>
        <footer><div><b>FCTX</b><p>AIを、触れられる可能性に。</p></div><div><a href="https://github.com/futurecortexlabs" target="_blank" rel="noopener noreferrer">GitHub ↗</a><a href="https://note.com/future_cortex" target="_blank" rel="noopener noreferrer">note ↗</a><a href="https://huggingface.co/FCTX" target="_blank" rel="noopener noreferrer">Hugging Face ↗</a></div><small>SAFE SIGHT / Interactive portfolio</small></footer>''')
        outputs = [output,kpis,table,raw,status,csv,report]
        def analyze(image,conf,equipment):
            return ppe.run(image,conf,equipment,False,demo.GRADIO_CACHE)
        def simulate(conf,equipment):
            return ppe.run(None,conf,equipment,True,demo.GRADIO_CACHE)
        sample.click(simulate,[confidence,required],outputs,concurrency_limit=1,concurrency_id="ppe",api_name=False)
        detect.click(analyze,[input_image,confidence,required],outputs,concurrency_limit=1,concurrency_id="ppe",api_name=False)
        for component in [input_image,confidence,required]:
            component.change(lambda: ppe.blank("入力・設定が変更されました。再解析またはサンプル体験を実行してください。"),outputs=outputs,concurrency_id="ppe",api_name=False)
        clear.click(lambda: [None,*ppe.blank()],outputs=[input_image,*outputs],concurrency_id="ppe",api_name=False)
        def make_brief(industry,source,needs,notes):
            text = f"PPE検出システム 開発相談\n\n現場：{industry}\n利用方法：{source}\n確認したい装備：{'、'.join(needs) or '未定'}\n課題：{notes.strip() or '相談しながら整理したい'}\n\n検討するステップ：実画像での検証 → 判定基準の整理 → 画面・出力・連携の実装\nメガネ・手袋・安全靴、動画・常時監視は個別開発の対象です。\n\nFCTX / SAFE SIGHT"
            Path(demo.GRADIO_CACHE).mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w",suffix=".txt",prefix="ppe-brief-",dir=demo.GRADIO_CACHE,delete=False,encoding="utf-8") as f:
                f.write(text)
                return text,f.name
        brief_button.click(make_brief,[industry,source,needs,notes],[brief,brief_file],api_name=False)
    return demo
