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
    POST /tts-gemini  {"text": "...", "voice_name": "Aoede", "rate": 0} → صوت (audio/wav|audio/mpeg)
        المحرك الذكي: تشكيل سياقي (بالسقوط الآلي) ثم صوت Gemini الأصلي عبر
        responseModalities=["AUDIO"] + speechConfig.voiceName (Aoede/Charon/Fenrir/Kore/Puck).
        الاستجابة بثّية (StreamingResponse) مع دعم Range (206) لفكّ TextStream لدى
        مشغّل Flutter والترجيع/التقديم بأمان. إن لم يكن مفتاحك يدعم نموذجاً صوتياً
        ينزل آلياً إلى دفق Edge (نفس /tts) دون كسر. header X-Tashkeel-Engine =
        gemini-audio|edge و X-Tashkeel-Diacrit = llm|onnx|raw.

بيئة التشغيل (Environment Variables على Render):
    GEMINI_API_KEY   مفتاح Google Gemini (إلزامي للمسار السياقي LLM؛ بدونه LLM=ONNX)
    GEMINI_MODEL     النموذج، افتراضي gemini-3.6-flash (عند مستخدم جديد 2.5-flash غير متاح — يرجع Google 404)
    GEMINI_TIMEOUT   مهلة الاستدعاء بالثواني، افتراضي 25
    GEMINI_MAX_CHARS حدّ أقصى لأحرف نص LLM، فوقه سقوط فوري إلى ONNX، افتراضي 1500
    TASHKEEL_TTS_VOICE  صوت النطق الافتراضي، افتراضي ar-EG-ShakirNeural
    TTS_ENABLED      true|false يعطل/يفعل /tts ، افتراضي true
    TTS_TIMEOUT      مهلة التوليف بالثواني، افتراضي 45
    EDGE_TTS_VOICE   صوت السقوط الآلي لمحرك Edge (تُنقل إليه أصوات Gemini وتحل محلها)، افتراضي ar-EG-ShakirNeural
    EDGE_PROSOBY_STYLE تفعيل/تعطيل القِدر الأبطأ (-12%) ودرجة الصوت عند نطق Edge (افتراضي true)
    GEMINI_AUDIO_MODEL  نموذج الصوت الذكي، افتراضي gemini-3.6-flash (عند عدم دعمه للصوت يُتخطى بكاش سلبي)
    GEMINI_AUDIO_VOICE  صوت Gemini الافتراضي، افتراضي Aoede
    GEMINI_AUDIO_TIMEOUT مهلة توليد صوت Gemini بالثواني، افتراضي 60
    PORT             (يضبطه Render تلقائياً)
"""
import asyncio
import base64
import json
import logging
import os
import re
import threading
import time
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

TTS_VOICE = os.environ.get("TASHKEEL_TTS_VOICE", "ar-EG-ShakirNeural").strip()
TTS_ENABLED = os.environ.get("TTS_ENABLED", "true").strip().lower() in ("1", "true", "yes")
TTS_TIMEOUT = int(float(os.environ.get("TTS_TIMEOUT", "45")))
EDGE_FALLBACK_VOICE = (
    os.environ.get("EDGE_TTS_VOICE", "ar-EG-ShakirNeural").strip() or "ar-EG-ShakirNeural"
)
EDGE_PROSOBY_STYLE = (
    os.environ.get("EDGE_PROSOBY_STYLE", "true").strip().lower() in ("1", "true", "yes")
)

# ── المحرك الذكي: صوت Gemini الأصلي (responseModalities AUDIO) ──
GEMINI_AUDIO_MODEL = (
    os.environ.get("GEMINI_AUDIO_MODEL", "gemini-3.6-flash")
    .strip()
    .replace("models/", "")
)
GEMINI_AUDIO_VOICE = os.environ.get("GEMINI_AUDIO_VOICE", "Aoede").strip()
GEMINI_AUDIO_TIMEOUT = float(os.environ.get("GEMINI_AUDIO_TIMEOUT", "60"))
# أصوات Gemini المدعومة رسمياً (للتحقق/التوثيق فقط؛ أي اسم يمرره جوجل يقبلها).
GEMINI_VOICES = {"Aoede", "Charon", "Fenrir", "Kore", "Puck"}

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


def _llm_request(payload):
    """ينفّذ طلباً واحداً لـ Gemini ويعيد dict الرد أو None (مع تفاصيل الخطأ).

    عند استنفاد الحصة (429) يقرأ ثانية واحدة بعد المهلة التي تشير إليها جوجل،
    ثم يُفعّل كولداون للطلبات التالية.
    """
    global _quota_blocked_until
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
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error(
            "Gemini رفض الطلب (HTTP %s) — تفاصيل جوجل:\n%s",
            exc.code,
            detail[:3000],
        )
        if exc.code == 429:
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", detail or "", re.I)
            delay = min(float(m.group(1)), 8.0) if m else 2.0
            logger.warning("استنفدت الحصة المجانية؛ إعادة المحاولة بعد %.1fs.", delay)
            time.sleep(delay)
            try:
                with urllib.request.urlopen(req, timeout=GEMINI_TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc2:  # noqa: BLE001
                logger.error("إعادة المحاولة بعد 429 فشلت: %s:%s",
                             type(exc2).__name__, exc2)
                _quota_blocked_until = time.monotonic() + _quota_cooldown
                logger.warning(
                    "حُجِبت استدعاءات LLM لمدة %ds بسبب استنداد الحصة المجانية.",
                    int(_quota_cooldown),
                )
                return None
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("خطأ اتصال بـ Gemini: %s:%s", type(exc).__name__, exc)
        return None


# تجميع نصوص متشابهة متكررة (التطبيق يطلب النص نفسه لاحقاً): ذاكرة مؤقتة بحدّ زمني.
_llm_cache = {}
_llm_cache_lock = threading.Lock()
_LLM_CACHE_TTL = 600.0

# ── كولداون الحصة المجانية (quota-blocked): حفظ 429 يمنع استدعاءات LLM مؤقتاً ──
_quota_blocked_until = 0.0
_quota_cooldown = 3600.0  # ساعة واحدة بعد استنفاد الحصة

# ── كاش LLM على القرص (JSON بسيط، يشحن مع start): ──
_LLM_DISK_PATH = os.environ.get("LLM_CACHE_PATH", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "llm_cache.json"
))


def _load_disk_cache():
    """يحمل كاش LLM من القرص عند بدء التشغيل (خيط آمن، يُستدعى مرة واحدة)."""
    try:
        with open(_LLM_DISK_PATH, "r", encoding="utf-8") as fh:
            pairs = json.load(fh)
        for raw_text, (_, diac) in pairs.items():
            _llm_cache[raw_text] = (time.monotonic(), diac)
        logger.info("تم شحن %d مدخلات LLM cache من القرص.", len(_llm_cache))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, KeyError):
        pass


def _save_disk_cache():
    """يحفظ الكاش على القرص (مجرد بسيط يُستدعى بانتظار)."""
    try:
        now = time.monotonic()
        with _llm_cache_lock:
            entries = {t: (v[0], v[1]) for t, v in _llm_cache.items()
                      if now - v[0] < _LLM_CACHE_TTL}
        with open(_LLM_DISK_PATH, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False, indent=0)
    except Exception:
        logger.debug("تعذّر حفظ كاش LLM على القرص.", exc_info=True)


_looks_diacritized_re = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def _looks_diacritized(text):
    """يكشف ما إذا كان النص يحتوي تشكيلًا كافياً (مثلاً أرسله التطبيق بعد تشكيل ONNX)."""
    words = text.split()
    if not words:
        return False
    marks = len(_looks_diacritized_re.findall(text))
    return marks >= len(words)


def _llm_quota_blocked():
    """إذا كانت الحصة منتهية، نتخطى Gemini مؤقتاً (ساعة واحدة مثلاً)."""
    if _quota_blocked_until > time.monotonic():
        return True
    return False


def _llm_cached(text):
    """LLM مع ذاكرة مؤقتة قصيرة لتفادي ضغط الحصة المجانية عند طلبات متطابقة."""
    now = time.monotonic()
    with _llm_cache_lock:
        hit = _llm_cache.get(text)
        if hit and now - hit[0] < _LLM_CACHE_TTL:
            return hit[1]
    out = _llm_diacritize(text)
    with _llm_cache_lock:
        if len(_llm_cache) > 512:
            _llm_cache.clear()
        _llm_cache[text] = (time.monotonic(), out)
    # حفظ على القرص عند وجود نتائج جديدة.
    try:
        _save_disk_cache()
    except Exception:
        pass
    return out


def _llm_diacritize(text):
    """استدعاء Gemini عبر REST (stdlib — بلا حزمة ثقيلة) وإرجاع النص المشكَّل."""
    payload = {
        "system_instruction": {"parts": [{"text": GEMINI_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }
    data = _llm_request(payload)
    if not data:
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
    if _llm_quota_blocked():
        logger.info("حصة Gemini محجوبة مؤقتاً (429)؛ نتخطى LLM إلى ONNX.")
        return None
    if len(text) > MAX_LLM_CHARS:
        logger.info("نص فوق %d حرفاً؛ نتخطى LLM إلى ONNX.", MAX_LLM_CHARS)
        return None
    if _looks_diacritized(text):
        logger.debug("نص مشكَّل بالكامل؛ نتخطى LLM (لا تُهدر الحصة).")
        return None
    fut = _llm_pool.submit(_llm_cached, text)
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

    نصٌّ شُكِّل سابقاً (مثل ما يرسله التطبيق بعد تشكيل ONNX) يمرّ كما هو
    بمرحلة "pass" دون أي استدعاء — يحفظ الحصة المجانية.

    تُرجع (النص المٌشكَّل، اسم المرحلة المستخدمة: llm|onnx|pass|raw).
    """
    if _looks_diacritized(text):
        return text, "pass"
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
#  — مسار السقوط الآلي فقط؛ لا يُمسّ مسار Gemini إطلاقاً.
# ════════════════════════════════════════════════════════════════

# تحسينات SSML لمحرك Edge لجعله أقرب إلى الإنسان (المسار الآلي فقط):
#   • فاصلة (، / ,)      → وقفة قصيرة ~200ms (نَفَس/تقسيم طبيعي)
#   • سطر جديد (\n)      → وقفة تنفّس ~450ms (شعر/فقرة)
#   • قِدر القراءة       → أبطأ قليلاً (-12%) كي لا تبدو العربية مشدودةً/آلية
#
# ملاحظة مُثبتة تجريبياً على edge-tts 7.2.8: الخدمة ترفض وسم <break/> في أي
# موضع (NoAudioReceived) مهما كانت صيغته (time/strength/quote). لذلك يُبني
# _build_ssml وثيقةَ <break/> الكاملة لمستعملي SSML الحقيقيين و/ssml، بينما
# يحقّق المسار المُرسل فعلياً (edge) الوقفاتَ بعلامات تحترمها النافثة:
# الفواصل تُبقيها كما هي (تضيف النافثة العصبية ~200ms) والأسطر الجديدة
# تُبقى فتراتِ جمل (~450ms عبر حد الجملة لدى الخدمة).
EDGE_PROSOBY_BASE_RATE_PCT = -12
EDGE_BREAK_PUNCT = '<break time="200ms"/>'
EDGE_BREAK_LINE = '<break time="450ms"/>'
_SSML_ESCAPE_TABLE = str.maketrans({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
})


def _rate_pct(value):
    """يقرأ معدّل المستخدم ("+20%"/"20"/0) ويُعيد عدداً صحيحاً."""
    try:
        return int(str(value).strip().rstrip("%").lstrip("+") or 0)
    except (TypeError, ValueError):
        return 0


def _combined_rate_pct(rate_pct):
    """يجمع المعدِّل الأساسي (-12%) مع معدِّل المستخدم (شريط السرعة)."""
    return EDGE_PROSOBY_BASE_RATE_PCT + _rate_pct(rate_pct)


def _ssml_inner(text):
    """يحوّل النص المشكَّل إلى محتوى SSML آمن: تهرّب XML ثم فواصل/أسطر.

    الترتيب حاسم: نهرّب الحروف أولاً ثم نُدخل وسوم <break/> الحقيقية (لا
    تُهرَّب لئلا تنقلب نصاً). عمليات قرص في الذاكرة حصراً — صفر تأخير.
    تُستخدم هذه في _build_ssml (وثيقة كاملة لمستعملي SSML).
    """
    safe = text.translate(_SSML_ESCAPE_TABLE)
    safe = safe.replace("،", EDGE_BREAK_PUNCT).replace(",", EDGE_BREAK_PUNCT)
    safe = re.sub(r"[ \t]*\n[ \t]*", EDGE_BREAK_LINE, safe)
    return safe


def _build_ssml(text, voice="ar-EG-ShakirNeural", rate_pct=0):
    """يبني وثيقة SSML كاملة محسّنة (وقفات + قِدر أبطأ قليلاً).

    الصيغة: <speak> → <voice> → <prosody> ← نص مهرَّب مع وسوم <break/>.
    مناسبة لمستعملي SSML الحقيقيين (كأوامر az/--ssml مثلًا)؛ أمّا مسار
    edge-tts فيحقّق الوقفاتَ ذاتها عبر علامات تحترمها الخدمة.
    """
    inner = _ssml_inner(text)
    total = _combined_rate_pct(rate_pct)
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="ar-SA"><voice name="{voice}">'
        f'<prosody rate="{total:+d}%" pitch="-0Hz">{inner}</prosody>'
        f"</voice></speak>"
    )


def build_ssml(text, voice, lang="ar-SA"):
    """يبني وثيقة SSML الأساسية (توافق مع عقد /ssml القديم دون تغيير)."""
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


def _edge_voice(voice):
    """يرحّل أي صوت إلى صوت Edge صالح، فتسلم الواجهة من أصوات Gemini.

    Edge لا يعرف Aoede/Charon/… فكان يفشل صامتاً (200 بلا صوت)؛
    كل مسار Edge يعبر هذا الاتجاه فيُسترجع صوت السقوط الآلي بدلًا منها.
    """
    v = (voice or "").strip() or TTS_VOICE
    if v in GEMINI_VOICES:
        logger.warning(
            "صوت Gemini (%s) لا يصلح لمحرك Edge؛ نستخدم (%s).", v, EDGE_FALLBACK_VOICE
        )
        return EDGE_FALLBACK_VOICE
    return v


def _edge_tts_communicate(text, voice, rate_pct=0, styled=True):
    """يولّد كائن edge-tts للنطق — صوت Edge صالح دائماً عبر _edge_voice.

    الصياغة مطابقة تماماً لما تتوقعه edge-tts 7.2.8: rate "^[+-]\d+%$"،
    pitch "^[+-]\d+Hz$"، volume "^[+-]\d+%$". styled=True يفعّل بنية
    _build_ssml على مستوى الصوت الفعلي (قِدر أبطأ -12% يجمع مع معدّل
    المستخدم + درجة صوت محايدة -0Hz)؛ styled=False يكتب الصيغة المخزنية
    (+0%/+0Hz) لإعادة محاولة آمنة. وقفات الفواصل والأسطر تبقى كما هي
    فتترجمها النافثة العصبية حكياً (~200ms/450ms) دون أي وسم يرفضه Edge.
    """
    import edge_tts

    voice = _edge_voice(voice)
    total = _combined_rate_pct(rate_pct) if styled else 0
    return edge_tts.Communicate(
        text,
        voice,
        rate=f"{total:+d}%",
        pitch="-0Hz" if styled else "+0Hz",
        volume="+0%",
        connect_timeout=10,
        receive_timeout=TTS_TIMEOUT,
    )


def _looks_like_mp3(data):
    """تحقق سريع من بنية MP3: غلاف ID3v2 أو رأس إطار MPEG (مزامنة 11 بت)."""
    if not data:
        return False
    if data[:3] == b"ID3":
        return True
    return (
        data[0] == 0xFF
        and (data[1] & 0xE0) == 0xE0
        and ((data[1] >> 3) & 0x03) in (1, 2, 3)
    )


async def _synthesize_async(text, voice, rate_pct):
    """يدفق الصوت من محرك Microsoft عبر edge-tts ويعيد MP3 bytes.

    محاولتان حذرتان: الأنماط المحسّنة أولاً (قِدر أبطأ + درجة صوت)، ثم
    المخزنية (+0%) احتياطاً — تُقبل النتيجة فقط إن صحّت بنيتها MP3.
    فلا يُسلَّم التطبيق MP3 صامتاً/مقطوعاً مع HTTP 200 تلو الآخر.
    """
    styles = (True, False) if EDGE_PROSOBY_STYLE else (False,)
    for styled in styles:
        label = "محسّنة" if styled else "مخزنية"
        buffer = bytearray()
        try:
            comm = _edge_tts_communicate(text, voice, rate_pct, styled=styled)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buffer.extend(chunk["data"])
        except Exception as exc:  # noqa: BLE001
            logger.error("استجابة edge-tts %s فاسدة (%s): %s",
                         label, type(exc).__name__, exc)
            continue
        audio = bytes(buffer)
        if _looks_like_mp3(audio):
            return audio
        logger.error("استجابة edge-tts %s ليست MP3 صالحة: %d بايت.",
                     label, len(audio))
    return None


def synthesize(text, voice, rate_pct=0):
    """توليف MP3 من نص (مشكَّل أو خام) مع مهلة قاتلة — فشل → None."""
    fut = _tts_pool.submit(
        lambda: asyncio.run(_synthesize_async(text, voice, rate_pct))
    )
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
#  المحرك الذكي: صوت Gemini الأصلي + بثّ استجابة مع Range
# ════════════════════════════════════════════════════════════════

# كاش سلبي لما يفشل صوت Gemini (404/غير مدعوم/مهلة): النموذج يُتجاهل مؤقتاً
# حتى لا نُهدر محاولتين بصوت فاشل في كل جملة بمحرك gemini.
_audio_neg_cache = {}
_audio_neg_lock = threading.Lock()
_AUDIO_NEG_COOLDOWN = float(os.environ.get("GEMINI_AUDIO_COOLDOWN", "900"))


def _audio_model_blocked():
    """بعد انتهاء كولداون العودة محاولة النموذج الصوتي. يعيد True إن كان محظوراً."""
    with _audio_neg_lock:
        until = _audio_neg_cache.get(GEMINI_AUDIO_MODEL, 0.0)
        if until > time.monotonic():
            return True
        if until:
            _audio_neg_cache.pop(GEMINI_AUDIO_MODEL, None)
        return False


def _block_audio_model(reason):
    """يسجّل فشل النموذج الصوتي ويمنعه مؤقتاً في الطلبات التالية."""
    with _audio_neg_lock:
        _audio_neg_cache[GEMINI_AUDIO_MODEL] = time.monotonic() + _AUDIO_NEG_COOLDOWN
    logger.warning(
        "نموذج الصوت %s يتعذّر (%s)؛ يُتخطى محاولات الصوت لـ %is…",
        GEMINI_AUDIO_MODEL,
        reason,
        int(_AUDIO_NEG_COOLDOWN),
    )


def _gemini_audio(text, voice_name):
    """يولّد صوتاً أَمثل من Gemini عبر responseModalities:["AUDIO"].

    يعيد (bytes, mimetype) عند النجاح وإلا None (ينسَكب الواجهة إلى Edge).
    لا يحاول النموذج أثناء الكولداون السلبي بعد فشل سابق.
    """
    if not GEMINI_API_KEY:
        return None
    if _audio_model_blocked():
        return None
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice_name}
                }
            },
        },
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_AUDIO_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=GEMINI_AUDIO_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error(
            "Gemini-Audio رفض الطلب (%s) بتفاصيل: %s", exc.code, detail[:1200]
        )
        if exc.code in (400, 403, 404, 429):
            _block_audio_model(f"HTTP {exc.code}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("خطأ اتصال بـ Gemini-Audio: %s:%s", type(exc).__name__, exc)
        return None
    try:
        part = data["candidates"][0]["content"]["parts"][0]
        inline = part.get("inlineData", {})
        mime = inline.get("mimeType", "")
        audio = base64.b64decode(inline.get("data", ""))
        if not audio or len(audio) < 100 or not mime.startswith("audio/"):
            logger.error(
                "Gemini-Audio استجابة فارغة أو غير صوتية: %d بايت (%s).",
                len(audio), mime,
            )
            return None
        return audio, mime
    except (KeyError, IndexError, TypeError):
        logger.error("استجابة Gemini-Audio بلا inlineData: %s",
                     json.dumps(data, ensure_ascii=False)[:400])
        return None


def _parse_range_header(raw, size):
    """يحلل Range: bytes=start-end | start- | -suffix ويعيد (start, end) أو None."""
    if not raw:
        return None
    m = re.match(r"bytes=(\d*)-(\d*)", raw.strip())
    if not m:
        return None
    start_s, end_s = m.group(1), m.group(2)
    if start_s == "" and end_s == "":
        return None
    if start_s == "":
        start = max(0, size - int(end_s))
        end = size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
        if start >= size:
            start = size
    end = min(end, size - 1)
    if start > end:
        return None
    return (start, end)


def _chunks(data, size=32768):
    for i in range(0, len(data), size):
        yield data[i:i + size]


def _audio_response(data, mimetype):
    """استجابة صوت بثّية مع دعم Range (206) — ليمكّن مشغّل Flutter من
    الترجيع/التقديم بأمان داخل ما تحمَّل (Buffered) حتى قبل اكتمال التنزيل."""
    length = len(data)
    headers = {"Accept-Ranges": "bytes"}
    rng = _parse_range_header(request.headers.get("Range"), length)
    if rng:
        start, end = rng
        part = data[start:end + 1]
        headers["Content-Range"] = f"bytes {start}-{end}/{length}"
        resp = Response(_chunks(part), status=206, mimetype=mimetype, headers=headers)
        resp.headers["Content-Length"] = str(len(part))
        return resp
    resp = Response(_chunks(data), mimetype=mimetype, headers=headers)
    resp.headers["Content-Length"] = str(length)
    return resp


def _edge_fallback_bytes(result, raw_text, rate_pct, voice=None):
    """سقوط البرنامج الصوتي: MP3 مُتحقَّق من النص المشكَّل ثم الخام.

    يُنطق النص المشكَّل أولاً؛ فإن تعذّر نُطق النص الخام كما جاء.
    الصوت دائمًا صوتُ Edge صالح (مرور Voice funnel عبر _edge_voice).
    يعيد (بايتات MP3 مستوفية أو None، الصوت الفعلي المستخدم).
    """
    voice = _edge_voice(voice)
    audio = synthesize(result, voice, rate_pct)
    if audio is None:
        logger.warning("تأخر/فشل نطق النص المشكَّل عبر Edge؛ ننطق النص الخام.")
        audio = synthesize(raw_text, voice, rate_pct)
    return audio, voice


def _parse_rate(value):
    """يقرأ معدّل المستخدم (رقم أو نص إنشائي "+20%") ويُعيد عدداً صحيحاً."""
    return _rate_pct(value)


def _tts_edge_bytes(text, voice, rate):
    """مسار Edge الكلاسيكي (تشكيل + نطق) ويعيد (بايتات، المحرك، الصوت الفعلي).

    الصوت المرتجَع صالحٌ دائماً لمحرك Edge (Voice funnel).
    """
    result, diac_engine = context_diacritize(text)
    audio, voice_used = _edge_fallback_bytes(result, text, rate, voice)
    final_engine = diac_engine if audio is not None else "raw"
    return audio, final_engine, voice_used


@app.route("/audio/tts", methods=["GET"])
def audio_tts_get():
    """نفس /tts لكن عبر GET — للبث المباشر لدى just_audio/ExoPlayer مع Range (206)."""
    if not TTS_ENABLED:
        return jsonify({"error": "tts disabled"}), 404
    text = (request.args.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    voice = _edge_voice(request.args.get("voice") or TTS_VOICE)
    rate = _parse_rate(request.args.get("rate", "0"))
    audio, engine, voice_used = _tts_edge_bytes(text, voice, rate)
    if audio is None:
        return jsonify({"error": "tts synthesis failed"}), 500
    resp = _audio_response(audio, "audio/mpeg")
    resp.headers["X-Tashkeel-Engine"] = engine
    resp.headers["X-Tashkeel-Diacrit"] = engine
    resp.headers["X-Voice"] = voice_used
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/audio/tts-gemini", methods=["GET"])
def audio_tts_gemini_get():
    """المحرك الذكي عبر GET — صوت Gemini عند جاهزيته، وإلا سقوط آلي
    إلى MP3 Edge مُتحقَّقٍ منه (Content-Length + Range 206) — لا يتوقف أبداً."""
    if not TTS_ENABLED:
        return jsonify({"error": "tts disabled"}), 404
    text = (request.args.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    voice = (request.args.get("voice_name") or GEMINI_AUDIO_VOICE).strip()
    rate = _parse_rate(request.args.get("rate", "0"))

    result, diac_engine = context_diacritize(text)
    audio_mime = _gemini_audio(result, voice)
    if audio_mime is not None:
        audio, mime = audio_mime
        resp = _audio_response(audio, mime)
        resp.headers["X-Tashkeel-Engine"] = "gemini-audio"
        resp.headers["X-Tashkeel-Diacrit"] = diac_engine
        resp.headers["X-Voice"] = voice
        resp.headers["Cache-Control"] = "no-store"
        return resp

    logger.warning("صوت Gemini غير متاح؛ نُستخدم محرك Edge.")
    audio, voice_used = _edge_fallback_bytes(result, text, rate, EDGE_FALLBACK_VOICE)
    if audio is None:
        return jsonify({"error": "tts synthesis failed"}), 500
    resp = _audio_response(audio, "audio/mpeg")
    resp.headers["X-Tashkeel-Engine"] = "edge"
    resp.headers["X-Tashkeel-Diacrit"] = diac_engine
    resp.headers["X-Voice"] = voice_used
    resp.headers["Cache-Control"] = "no-store"
    return resp


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
    voice = _edge_voice(data.get("voice") or TTS_VOICE)
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


@app.route("/tts-gemini", methods=["POST"])
def tts_gemini_endpoint():
    """المحرك الذكي: تشكيل سياقي ← صوت Gemini الأصلي (AUDIO) — بثّ مع Range.

    body: {"text": "...", "voice_name": "Aoede|Charon|Fenrir|Kore|Puck", "rate": 0}
    عند فشل/غياب صوت Gemini ← سقوط آلي إلى MP3 Edge مُتحقَّقٍ منه
    (Content-Length + Range 206) — لا يتوقف أبداً.
    الرؤوس: X-Tashkeel-Engine = gemini-audio|edge ، X-Tashkeel-Diacrit ، X-Voice.
    """
    if not TTS_ENABLED:
        return jsonify({"error": "tts disabled"}), 404
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    voice = (data.get("voice_name") or GEMINI_AUDIO_VOICE).strip()
    try:
        rate = f"{int(data.get('rate', 0)):+d}%"
    except (TypeError, ValueError):
        rate = "+0%"

    # 1) تشكيل سياقي (نص كامل) مع سقوط آلي حتى الخام.
    result, diac_engine = context_diacritize(text)

    # 2) جرّب صوت Gemini الأصلي (نموذج صوتي إن كان مفعّلاً لمفتاحك).
    audio_mime = _gemini_audio(result, voice)
    if audio_mime is not None:
        audio, mime = audio_mime
        resp = _audio_response(audio, mime)
        resp.headers["X-Tashkeel-Engine"] = "gemini-audio"
        resp.headers["X-Tashkeel-Diacrit"] = diac_engine
        resp.headers["X-Voice"] = voice
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # 3) سقوط آلي: MP3 Edge مُتحقَّقٍ منه من النص المشكَّل (ثم الخام إن تعثر).
    logger.warning("صوت Gemini غير متاح؛ نُستخدم محرك Edge.")
    audio, voice_used = _edge_fallback_bytes(result, text, rate, EDGE_FALLBACK_VOICE)
    if audio is None:
        return jsonify({"error": "tts synthesis failed"}), 500
    resp = _audio_response(audio, "audio/mpeg")
    resp.headers["X-Tashkeel-Engine"] = "edge"
    resp.headers["X-Tashkeel-Diacrit"] = diac_engine
    resp.headers["X-Voice"] = voice_used
    resp.headers["Cache-Control"] = "no-store"
    return resp
    return resp


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# شحن كاش LLM من القرص عند بدء التشغيل (لا يُهدر الحصة على النصوص المتكررة).
_load_disk_cache()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)