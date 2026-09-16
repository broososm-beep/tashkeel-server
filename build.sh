#!/usr/bin/env bash
# يُشغَّل تلقائياً عند البناء على Render (Build Command) لتجهيز كُل المحركين.
# 1) تثبيت متطلبات النظام + بايثون.
# 2) تنزيل نموذج CATT (Apache-2.0) مسبقاً إلى catt_tashkeel/onnx_models/
#    حتى لا يُنبَّه عند أول طلب — وأَي تعذُّر لا يوقف البناء (يُستكمل بعدئذٍ).
set -u

pip install -r requirements.txt

# تنزيل نموذج CATT EO (~74MB) — الوضع الفعلي على Render: قرصٌ مؤقت يُملأ عند البناء.
MODEL_DIR="catt_tashkeel/onnx_models/eo_model"
if [ ! -f "$MODEL_DIR/encoder.onnx" ]; then
  echo "[build.sh] Downloading CATT EO model (~74MB)..."
  mkdir -p "$MODEL_DIR"
  ZIP="catt_tashkeel/onnx_models/eo_model_onnx.zip"
  curl -fsSL -o "$ZIP" "https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip" \
    && unzip -q -o "$ZIP" -d "$MODEL_DIR" \
    && rm -f "$ZIP" \
    && echo "[build.sh] CATT model ready." \
    || echo "[build.sh] CATT model download failed (will be lazy-downloaded at first request)."
else
  echo "[build.sh] CATT model already present."
fi