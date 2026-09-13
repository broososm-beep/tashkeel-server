# -*- coding: utf-8 -*-
"""
خادم التشكيل (tashkeel_server) — إصدار المحرك الخفيف (FR-18 C)
العقد الإلزامي (لا يتغير):
    POST /tashkeel
      الجسم:  {"text": "<نص عربي خام>"}
      الرد:   {"diacritized": "<نص مُشكَّل>"}   (كود 200)
أي فشل داخلي يُرجع كود 500 مع رسالة مجردة (بدون محتوى النص نفسه).

المحرك: text2tashkeel (ONNX عبر onnxruntime) — عجلة صغيرة ~10MB، بلا
PyTorch/TensorFlow، ولا شبكة أثناء التشغيل. يحل انهيار الذاكرة (OOM)
الذي سببه المحرك السابق على خطط Render المجانية (512MB).
"""
import os
import threading

from flask import Flask, request, jsonify

app = Flask(__name__)

_diacritizer = None
_di_lock = threading.Lock()
_warmup_started = threading.Event()


def get_diacritizer():
    """يحمّل محرك التشكيل مرة واحدة فقط (خيط آمن) ثم يعيد استخدامه."""
    global _diacritizer
    if _diacritizer is None:
        with _di_lock:
            if _diacritizer is None:
                from text2tashkeel import Diacritizer
                _diacritizer = Diacritizer()
    return _diacritizer


def _warmup():
    """تهيئة مسبقة عند الإقلاع كي يكون أول طلب فوريًا (خارج مهلة الطلب)."""
    if _warmup_started.is_set():
        return
    _warmup_started.set()
    try:
        get_diacritizer().diacritize("بسم الله الرحمن الرحيم")
    except Exception:
        pass


def is_arabic_text(text):
    """يتحقق من احتواء النص على حروف عربية (لا يشكّل النص الإنجليزي)."""
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


@app.route("/tashkeel", methods=["POST"])
def tashkeel():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    if not is_arabic_text(text):
        # نص بلا حروف عربية: يُعاد كما هو دون تشكيل (مكافئ FR-4)
        return jsonify({"diacritized": text})
    try:
        result = get_diacritizer().diacritize(text)
        return jsonify({"diacritized": result})
    except Exception:
        return jsonify({"error": "diacritization failed"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# تهيئة مسبقة خلفية عند إقلاع gunicorn (لا تحجب الخادم)
if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
    threading.Thread(target=_warmup, daemon=True).start()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)