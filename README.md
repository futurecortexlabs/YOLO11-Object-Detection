---
title: FCTX | 画像AI・物体検出
emoji: 🔎
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: "5.33.0"
python_version: "3.12"
app_file: app.py
pinned: false
short_description: 画像から物体を検出・可視化。FCTXの画像AI開発を体験。
---

# FCTX | 画像AI・物体検出

画像をアップロードすると、事前学習済みYOLO11nで物体検出を行う体験型ポートフォリオです。
FCTXのアプリ実装・可視化・データ出力を体験し、開発相談につなげる構成です。

## 実装済み

- COCO 80クラス（人物・車・動物・日用品など）の物体検出
- アップロード、Ultralytics同梱の写真2枚によるサンプル体験（実推論）
- 信頼度・IoU設定、検出枠つき画像の保存
- 検出数・種類数・推論時間、座標テーブル、UTF-8 BOM付きCSV
- 画像・設定変更時の結果クリア、リセット、未入力・検出なし・失敗の案内
- スマートフォン対応、開発サービス・FAQ・相談メモ・Googleフォームへの導線
- 10MBのアップロード制限、2,000万画素の検証、長辺1920pxへの縮小
- 推論の直列化、待機上限10件、一時キャッシュの期限削除

## Hugging Face Spacesへの配置

1. Hugging FaceでSpaceを作成し、SDKに **Gradio** を選択します。
2. 以下の6ファイルをSpaceリポジトリのルートへアップロードします。
   - `app.py`
   - `vision_ui.py`
   - `style.css`
   - `requirements.txt`
   - `packages.txt`
   - `README.md`
3. ビルド完了後、Appタブでサンプル画像を選び「物体検出を実行」を押します。
4. 結果画像・CSV・相談リンクを確認します。

`dist/fctx-huggingface.zip` がある場合は展開して上記ファイルを配置してください。ZIPのままアップロードしても起動しません。
モデル重みは初回推論時にUltralyticsから取得します。トークン不要。CPUで動作します。
初回のモデル取得やSpaceのスリープ復帰には時間がかかります。ハードウェア・料金条件はHugging Face上で確認してください。
`0.0.0.0:7860`で起動します。SpacesではPORTを変更する必要はありません。
`packages.txt`はOpenCVのLinuxランタイムライブラリを追加します。

このフォルダの旧PPE関連ファイルは今回のSpaceでは使用しません。`.venv`、`models`、`*.pt`、テスト、ローカル実行スクリプトは配置不要です。

## ローカル実行

Python 3.12を使用します。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Linux/macOSでは`.venv/bin/python`に読み替えます。http://localhost:7860 を開いてください。

## 表記・相談先の変更

- `vision_ui.py` のCONTACT：相談フォームURL
- `vision_ui.py`：ブランド名、説明、プロフィールURL
- `style.css`：配色・余白・モバイル表示
- `README.md`先頭のtitle：Spacesで表示するタイトル

## データと検出範囲

画像はサーバーに送信して推論します。アプリは画像を学習に使用しません。
入力・出力・CSVは一時保存し、稼働中は1時間を超えたキャッシュを10分間隔で削除します。
ホスティング側のログやバックアップは別途管理してください。機密・個人情報を含む写真は送信しないでください。
縮小した場合の座標は出力画像基準です。推論時間にはモデル読込、待機、画像描画時間は含みません。
信頼度は正解率の保証ではありません。業務導入前に現場データで検証してください。
傷・不良品・ヘルメットなどの専用検出、動画・カメラ監視、通知、履歴DBは今回の実装範囲外です。

## モデルと利用条件

YOLO11nのCOCO事前学習済み重みを利用します。FCTX独自学習のモデルではありません。
サンプル写真はUltralyticsパッケージ同梱のbus.jpgとzidane.jpgです。
Ultralyticsのコード・モデル・アセットの利用条件は公式ライセンスを確認し、公開・商用提供時の条件に従ってください。

- [YOLO11公式ドキュメント](https://docs.ultralytics.com/models/yolo11/)
- [Ultralyticsライセンス](https://www.ultralytics.com/license)
- [Spaces設定](https://huggingface.co/docs/hub/spaces-config-reference)
- [Spaces依存関係](https://huggingface.co/docs/hub/spaces-dependencies)

## 検証

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_app.py -v
```

Windowsのローカル実行確認と、Hugging Face Linux上でのビルド・実行確認は別です。Space公開後に必ずサンプル推論を確認してください。

[開発相談](https://docs.google.com/forms/d/e/1FAIpQLSeeW9u8w2cvZnnRlUIRfXD-mvKGBvxhFouMYS4FwKCDfhhw4w/viewform?usp=header) / [GitHub](https://github.com/futurecortexlabs) / [note](https://note.com/future_cortex) / [Hugging Face](https://huggingface.co/FCTX)
