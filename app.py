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
import logging
import os
import threading
import traceback

from flask import Flask, request, jsonify

# تقييد خيوط onnxruntime إلى خيط واحد (أذكى مع gunicorn وأخفّ ذاكرة).
os.environ.setdefault("TT_ORT_THREADS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tashkeel")

app = Flask(__name__)

_diacritizer = None
_di_lock = threading.Lock()


def get_diacritizer():
    """يحمّل محرك التشكيل مرة واحدة فقط (خيط آمن) ثم يعيد استخدامه.

    التحميل كسول عند أول طلب عربي (لا خيط خلفي عند الإقلاع، فلا تُبنى جلسة
    onnxruntime خلف gunicorn fork).
    """
    global _diacritizer
    if _diacritizer is None:
        with _di_lock:
            if _diacritizer is None:
                try:
                    from text2tashkeel import Diacritizer
                    _di = Diacritizer()
                    _diacritizer = _di
                except Exception:
                    logger.error("فشل تحميل محرك التشكيل:\n%s", traceback.format_exc())
                    raise
                # تهيئة مسبقة خفيفة: أول استدعاء يبني جلسة ORT ويخزنها.
                # أي خطأ هنا مؤقت والطلبات التالية ستنجح.
                try:
                    _di.diacritize("بسم الله")
                except Exception:
                    logger.warning("تهيئة محرك التشكيل (قد تنجح الطلبات التالية):\n%s",
                                   traceback.format_exc())
    return _diacritizer


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
        logger.error("فشل عملية التشكيل:\n%s", traceback.format_exc())
        return jsonify({"error": "diacritization failed"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)