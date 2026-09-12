---
title: Future Cortex Labs - Image AI Demo
emoji: 🔍
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "5.33.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Future Cortex Labs | 画像AI開発デモ

画像AIの開発力を実際に体験できる、相談導線つきポートフォリオです。
既存の相談フォーム・GitHub・note・Hugging Faceへのリンクを使用しています。

## 機能

- YOLO11n事前学習済みモデルによるCOCO 80クラスの物体検出
- アップロード、信頼度・IoU設定、検出枠つき画像の保存
- 検出数・種類数・推論時間、座標テーブル、UTF-8 BOM付きCSV
- 入力なし・検出なし・処理失敗への案内、リセット
- ファイル上限10MB、2,000万画素、長辺1920pxへ縮小
- モデルの遅延ロードとキャッシュ、推論の直列化、待機上限10件
- モバイル対応の紹介ページ、サービス説明、開発の流れ、FAQ、相談CTA

## 起動

Python 3.10〜3.12を利用してください。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

macOS / Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

http://localhost:7860 を開きます。初回推論時にUltralyticsのモデル重みを取得します。
インターネット接続とプロジェクトへの書込権限が必要です。PORT環境変数でポート変更可。

## Hugging Face Spaces

Gradio SDKのSpaceへapp.py、style.css、requirements.txt、このREADMEを配置します。
CPUでも動作します。初回モデル取得・スリープからの復帰には時間がかかります。
本変更はデプロイや公開を自動では実施しません。

## カスタマイズ

- app.pyのCONTACT: 相談フォームURL
- app.pyのHTML: ブランド、サービス紹介、各種プロフィールリンク
- style.css: 配色・余白・レスポンシブ表示

## データと検出の範囲

画像はサーバーへ送信して推論します。アプリから学習には利用しません。
Gradioの画像・CSVキャッシュは1時間を超えたものを10分間隔で削除します。
ホスティング側のログ・バックアップは別途管理してください。
縮小した場合、CSVの座標は出力画像基準です。信頼度は0〜1のモデル出力で、正解率の保証ではありません。
傷・欠陥検出などの独自対象、動画、カメラ、外部システム連携はこのデモには未実装です。

## 確認

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

モデル推論についてはUltralytics公式の[Predictドキュメント](https://docs.ultralytics.com/modes/predict/)を参照。
公開・商用運用に用いるモデルとライブラリの利用条件は[Ultralytics公式](https://www.ultralytics.com/license)で確認してください。

[開発相談](https://forms.gle/SaWGZFu8J7DgbytL7) / [GitHub](https://github.com/futurecortexlabs) / [note](https://note.com/future_cortex) / [Hugging Face](https://huggingface.co/FCTX)
