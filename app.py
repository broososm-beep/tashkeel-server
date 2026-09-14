# -*- coding: utf-8 -*-
"""
خادم التشكيل + النطق (tashkeel_server) — إصدار المعالجة السياقية (FR-18 C, FR-19)

العقد الإلزامي القديم (لا يتغير — تطبيق الموبايل الحالي يعمل عليه دون أي كسر):
    POST /tashkeel  {"text": "..."}                      → {"diacritized": "..."}   (200)
    GET  /health                                          → {"status": "ok"}          (200)

Endpoints جديدة (إضافية، اختيارية):
    POST /tashkeel/context  {"text": "..."}               → {"diacritized": "...", "engine": "llm|onnx|raw"} (200)
        معالجة سياقية كاملة: النص كاملاً يذهب إلى Gemini بنظام-تعليمات صارم (إعراب،
        التقاء ساكنين، ضرورات شعرية). عند غياب المفتاح/المهلة/الفشل → سقوط آلي إلى
        محرك ONNX المحلي ثم إلى النص الخام. لن يتوقف أبداً.
    POST /ssml  {"text": "...", "voice": "..."}           → {"ssml": "<speak>...</speak>", "engine": "..."} (200)
        يُرجع فقط وثيقة SSML (لكل عميل يدعمها).
    POST /tts  {"text": "...", "voice": "...", "rate": 0} → ملف audio/mpeg (MP3)
        خط أنابيب كامل: تشكيل سياقي (بالسقوط الآلي) ثم نطق عبر محرك مايكروسوفت
        (edge-tts / Edge Neural). إن تأخر أو فشل التشكيل يُنطَق النص الخام —
        لا يتوقف التطبيق أبداً. header X-Tashkeel-Engine يوضح المرحلة المستخدمة.

بيئة التشغيل (Environment Variables على Render):
    GEMINI_API_KEY   مفتاح Google Gemini (إلزامي للمسار السياقي LLM؛ بدونه LLM=ONNX)
    GEMINI_MODEL     النموذج، افتراضي gemini-3.6-flash (عند مستخدم جديد 2.5-flash غير متاح — يرجع Google 404)
    GEMINI_TIMEOUT   مهلة الاستدعاء بالثواني، افتراضي 25
    GEMINI_MAX_CHARS حدّ أقصى لأحرف نص LLM، فوقه سقوط فوري إلى ONNX، افتراضي 1500
    TASHKEEL_TTS_VOICE  صوت النطق الافتراضي، افتراضي ar-SA-HamedNeural
    TTS_ENABLED      true|false يعطل/يفعل /tts ، افتراضي true
    TTS_TIMEOUT      مهلة التوليف بالثواني، افتراضي 45
    PORT             (يضبطه Render تلقائياً)
"""
import asyncio
import json
import logging
import os
import threading
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from flask import Flask, Response, jsonify, request

# تقييد خيوط onnxruntime إلى خيط واحد (أذكى مع gunicorn وأخفّ ذاكرة).
os.environ.setdefault("TT_ORT_THREADS", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("tashkeel")

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────
#  الإعدادات من بيئة التشغيل (متغيرات Render)
# ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = (
    os.environ.get("GEMINI_MODEL", "gemini-3.6-flash").strip().replace("models/", "")
)
GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "25"))
MAX_LLM_CHARS = int(os.environ.get("GEMINI_MAX_CHARS", "1500"))

TTS_VOICE = os.environ.get("TASHKEEL_TTS_VOICE", "ar-SA-HamedNeural").strip()
TTS_ENABLED = os.environ.get("TTS_ENABLED", "true").strip().lower() in ("1", "true", "yes")
TTS_TIMEOUT = int(float(os.environ.get("TTS_TIMEOUT", "45")))

# تعليمات صارمة للمعالجة السياقية (System Prompt معتمد في الطلب إلى Gemini).
GEMINI_SYSTEM_PROMPT = (
    "أنت مولّد تشكيل عربي فصحى بإتقان عالٍ. استقبل النص العربي المضمَّن كله على "
    "أنه نص واحد متكامل السياق (لا تتعامل معه كلمةً كلمةً)، وأعده مشكَّلًا تشكيلًا "
    "كاملاً مضبوطًا بالضوابط الآتية:\n"
    "1) الإعراب السليم: ارفع المبتدأ والخبر والفاعل ونائبه، وانصب المفعول به وما "
    "يُعمل فيه النصب، واجرر المضاف إليه والمجرورات، ووفّق الفعل المضارع بين رفع "
    "ونصب وجزم حسب ما يسبقه.\n"
    "2) قواعد التقاء الساكنين في الفصحى والشعر: إن اجتمع ساكنان فاكسر ما قبل "
    "الساكن الأخير كي لا يلتبس النطق، مثل: «لَمْ يَعْبُدِ».\n"
    "3) الضرورات الشعرية: إذا كان النص شِعراً أو سجعاً أو موزوناً، فالتزم الوزن "
    "وتحمل الضرورة الشعرية بالسكون أو التقليل من الحُرَك الثقال حيث يوجب الوزن، "
    "مثل تسكين الهاء في «فَهْوَ»، دون كسر الوزن أو المعنى.\n"
    "4) لا تقدّم أي شرح ولا جملات إنشائية ولا أسطراً زائدة ولا علامات اقتباس حول "
    "النص: الناتج الوحيد هو النص نفسه بعد تشكيله.\n"
    "5) حافظ على الإملاء والنقاط والفواصل وعدد الكلمات كما هي، ولا تُبدّل الكلمات "
    "ولا تُضف ولا تُسقط.\n"
    "أقتصر الجواب على النص المُشكَّل فقط."
)

# تجمعان عشرة-الوزن للاستدعاء الشارد: استدعاء Gemini، وتوليف الصوت.
_llm_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")
_tts_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")

_diacritizer = None
_di_lock = threading.Lock()


def get_diacritizer():
    """يحمّل محرك التشكيل المحلي (ONNX خفيف) مرة واحدة فقط (خيط آمن)."""
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
                try:
                    _di.diacritize("بسم الله")
                except Exception:
                    logger.warning("تهيئة محرك التشكيل (قد تنجح الطلبات التالية):\n%s",
                                   traceback.format_exc())
    return _diacritizer


def is_arabic_text(text):
    """يتحقق من احتواء النص على حروف عربية (لا يشكّل النص الإنجليزي)."""
    return any("\u0600" <= ch <= "\u06FF" for ch in text)


# ════════════════════════════════════════════════════════════════
#  المعالجة السياقية (Context-Aware Pipeline) — Gemini ثم سقوط آلي
# ════════════════════════════════════════════════════════════════


def _llm_diacritize(text):
    """استدعاء Gemini عبر REST (stdlib — بلا حزمة ثقيلة) وإرجاع النص المشكَّل."""
    payload = {
        "system_instruction": {"parts": [{"text": GEMINI_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error(
            "Gemini رفض الطلب (HTTP %s) — تفاصيل جوجل:\n%s",
            exc.code,
            detail[:3000],
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("خطأ اتصال بـ Gemini: %s:%s", type(exc).__name__, exc)
        return None
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return None


def _llm_diacritize_safe(text):
    """LLM مع مهلة زمنية صارمة؛ أي فشل/مهلة → None (فيُفعَّل السقوط الآلي)."""
    if not GEMINI_API_KEY:
        logger.info("لا يوجد GEMINI_API_KEY؛ نتخطى المسار السياقي إلى ONNX.")
        return None
    if len(text) > MAX_LLM_CHARS:
        logger.info("نص فوق %d حرفاً؛ نتخطى LLM إلى ONNX.", MAX_LLM_CHARS)
        return None
    fut = _llm_pool.submit(_llm_diacritize, text)
    try:
        out = fut.result(timeout=GEMINI_TIMEOUT + 8)
    except TimeoutError:
        logger.warning("Gemini تجاوز المهلة (%ss)؛ ننتقل إلى البديل.", GEMINI_TIMEOUT)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("استدعاء Gemini فشل: %s:%s", type(exc).__name__, exc)
        return None
    if not out:
        return None
    # إزالة أي إطار Markdown قد يضعه النموذج احتياطاً.
    if out.startswith("```"):
        out = out.strip("`")
        if "\n" in out:
            out = out.split("\n", 1)[1]
    return out.strip() or None


def context_diacritize(text):
    """سلسلة السقوط الآلي: Gemini→ONNX→النص الخام. لا تعود None أبداً.

    تُرجع (النص المٌشكَّل، اسم المرحلة المستخدمة: llm|onnx|raw).
    """
    llm_out = _llm_diacritize_safe(text)
    if llm_out:
        return llm_out, "llm"
    try:
        return get_diacritizer().diacritize(text), "onnx"
    except Exception:
        logger.error("فشل المحرك المحلي:\n%s", traceback.format_exc())
    return text, "raw"


# ════════════════════════════════════════════════════════════════
#  SSML + النطق عبر محرك مايكروسوفت (edge-tts)
# ════════════════════════════════════════════════════════════════


def build_ssml(text, voice, lang="ar-SA"):
    """يبني وثيقة SSML متكاملة من النص المشكَّل (مع تهرّب آمن)."""
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="{lang}">'
        f'<voice name="{voice}">{safe}</voice>'
        "</speak>"
    )


async def _synthesize_async(text, voice, rate):
    """يدفق الصوت من محرك Microsoft عبر edge-tts ويعيد MP3 bytes."""
    import edge_tts

    comm = edge_tts.Communicate(
        text, voice, rate=rate, connect_timeout=10, receive_timeout=TTS_TIMEOUT
    )
    buffer = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            buffer.extend(chunk["data"])
    return bytes(buffer)


def synthesize(text, voice, rate):
    """توليف MP3 من نص (مشكَّل أو خام) مع مهلة قاتلة — فشل → None."""
    fut = _tts_pool.submit(lambda: asyncio.run(_synthesize_async(text, voice, rate)))
    try:
        audio = fut.result(timeout=TTS_TIMEOUT + 12)
        return audio if audio else None
    except TimeoutError:
        logger.error("تأخرت عملية النطق أكثر من %ss؛ نُعيد فشلاً للاسترجاع.", TTS_TIMEOUT)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("فشل النطق: %s:%s", type(exc).__name__, exc)
        return None


# ════════════════════════════════════════════════════════════════
#  نقاط النهاية
# ════════════════════════════════════════════════════════════════


@app.route("/tashkeel", methods=["POST"])
def tashkeel():
    """العقد الإلزامي القديم — دون أي تغيير (توافق كامل مع تطبيق الموبايل)."""
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


@app.route("/tashkeel/context", methods=["POST"])
def tashkeel_context():
    """معالجة سياقية كاملة: النص كاملاً → Gemini (تعليمات صارمة) مع سقوط آلي."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    if not is_arabic_text(text):
        return jsonify({"diacritized": text, "engine": "pass"})
    result, engine = context_diacritize(text)
    return jsonify({"diacritized": result, "engine": engine})


@app.route("/ssml", methods=["POST"])
def ssml_endpoint():
    """يُعيد فقط وثيقة SSML (لكل عميل يدعمها). نص إنجليزي → بدون تشكيل."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    voice = (data.get("voice") or TTS_VOICE).strip()
    result, engine = context_diacritize(text)
    return jsonify({"ssml": build_ssml(result, voice), "engine": engine, "voice": voice})


@app.route("/tts", methods=["POST"])
def tts_endpoint():
    """خط أنابيب كامل: تشكيل سياقي → SSML → نطق Microsoft → MP3.

    السقوط الآلي: إن تأخر أو فشل التشكيل يُنطَق النص الخام، فلا يتوقف التطبيق.
    الرؤوس: X-Tashkeel-Engine = llm|onnx|raw ، X-Voice المستخدم.
    """
    if not TTS_ENABLED:
        return jsonify({"error": "tts disabled"}), 404
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    voice = (data.get("voice") or TTS_VOICE).strip()
    try:
        rate = f"{int(data.get('rate', 0)):+d}%"
    except (TypeError, ValueError):
        rate = "+0%"

    # 1) تشكيل سياقي (نص كامل) مع سقوط آلي حتى الخام.
    result, engine = context_diacritize(text)

    # 2) توليف الصوت من النص المشكَّل.
    audio = synthesize(result, voice, rate)
    if audio is None:
        # 3) fallback نهائي: النص الخام كما هو (ضمان عدم التوقف).
        logger.warning("تأخر/فشل نطق النص المشكَّل؛ ننطق النص الخام.")
        engine = "raw"
        audio = synthesize(text, voice, rate)
    if audio is None:
        return jsonify({"error": "tts synthesis failed"}), 500

    return Response(
        audio,
        mimetype="audio/mpeg",
        headers={
            "X-Tashkeel-Engine": engine,
            "X-Voice": voice,
            "Cache-Control": "no-store",
        },
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)