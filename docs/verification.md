# 動作確認記録

- Windows / Python 3.12 / Gradio 5.33.0
- unittest: 4件成功（入力なし・RGB変換/縮小/CSV・検出なし・例外時のクリア）
- pip check: 依存関係の不整合なし
- YOLO11n + Ultralytics同梱bus.jpg: 5個・2種類を検出
- Edgeブラウザ: 未入力案内、画像アップロード、実モデル推論、CSVリンク、リセットを確認
- デスクトップ1440px / モバイル390px: 表示確認、ページ幅のはみ出しなし
- docs/desktop.png、docs/mobile.png: 検証時の画面キャプチャ

外部フォームへの送信、公開環境へのデプロイは実施していません。
