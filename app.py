# -*- coding: utf-8 -*-
"""
خادم التشكيل (tashkeel_server) — خدمة خارجية صغيرة
العقد الإلزامي (لا يتغير):
    POST /tashkeel
      الجسم:  {"text": "<نص عربي خام>"}
      الرد:   {"diacritized": "<نص مُشكَّل>"}   (كود 200)
أي فشل داخلي يُرجع كود 500 مع رسالة مجردة (بدون محتوى النص نفسه).
"""
import os

from flask import Flask, request, jsonify

app = Flask(__name__)

_diacritizer = None


def get_diacritizer():
    """يُحمّل محرك التشكيل مرة واحدة فقط ثم يعيد استخدامه."""
    global _diacritizer
    if _diacritizer is not None:
        return _diacritizer
    try:
        # الواجهة الحديثة لحزمة arabic-diacritizer
        from diacritize import Diacritizer
        _diacritizer = Diacritizer.from_pretrained()
    except Exception:
        # الواجهة القديمة من نفس الحزمة
        from arabic_diacritizer import ArabicDiacritizer
        _diacritizer = ArabicDiacritizer()
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
        # نص بلا حروف عربية: يُعاد كما هو دون تشكيل (FR-4)
        return jsonify({"diacritized": text})
    try:
        result = get_diacritizer().diacritize(text)
        return jsonify({"diacritized": result})
    except Exception:
        return jsonify({"error": "diacritization failed"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)