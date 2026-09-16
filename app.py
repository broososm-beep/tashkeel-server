# -*- coding: utf-8 -*-
"""
خادم التشكيل + النطق (tashkeel_server) — إصدار المعالجة السياقية (FR-18 C, FR-19)

العقد الإلزامي القديم (لا يتغير — تطبيق الموبايل الحالي يعمل عليه دون أي كسر):
    POST /tashkeel  {"text": "..."}                      → {"diacritized": "..."}   (200)
    GET  /health                                          → {"status": "ok"}          (200)

Endpoints جديدة (إضافية، اختيارية):
    POST /tashkeel/context  {"text": "..."}               → {"diacritized": "...", "engine": "llm|catt|onnx|raw"} (200)
        معالجة سياقية كاملة: النص كاملاً يذهب إلى Gemini بنظام-تعليمات صارم (إعراب،
        التقاء ساكنين، ضرورات شعرية). عند غياب المفتاح/المهلة/الفشل → سقوط آلي إلى
        CATT (محرك ONNX محلي دقيق) ثم إلى محرك ONNX ثم النص الخام. لن يتوقف أبداً.
    POST /tts-gemini  {"text": "...", "voice_name": "Aoede", "rate": 0} → صوت (audio/wav|audio/mpeg)
        المحرك الذكي: تشكيل سياقي (بالسقوط الآلي) ثم صوت Gemini الأصلي عبر
        responseModalities=["AUDIO"] + speechConfig.voiceName (Aoede/Charon/Fenrir/Kore/Puck).
        الاستجابة بثّية (StreamingResponse) مع دعم Range (206) لفكّ TextStream لدى
        مشغّل Flutter والترجيع/التقديم بأمان. إن لم يكن مفتاحك يدعم نموذجاً صوتياً
        ينزل آلياً إلى دفق Edge (نفس /tts) دون كسر. header X-Tashkeel-Engine =
        gemini-audio|edge و X-Tashkeel-Diacrit = llm|catt|onnx|raw.

بيئة التشغيل (Environment Variables على Render):
    GEMINI_API_KEY   مفتاح Google Gemini (إلزامي للمسار السياقي LLM؛ بدونه LLM=ONNX)
    GEMINI_MODEL     النموذج، افتراضي gemini-3.6-flash (عند مستخدم جديد 2.5-flash غير متاح — يرجع Google 404)
    GEMINI_TIMEOUT   مهلة الاستدعاء بالثواني، افتراضي 25
    GEMINI_MAX_CHARS حدّ أقصى لأحرف نص LLM، فوقه سقوط فوري إلى ONNX، افتراضي 1500
    CATT_MAX_CHARS   حدّ CATT لكل قطعة ONNX، افتراضي 1024 (سقف النموذج)
    CATT_TIMEOUT     مهلة CATT بالثواني، افتراضي 60
    CATT_MIN_LEN     حدّ أدنى لعدد علامات تشكيل CATT كي تُقبل نتيجته، افتراضي 0
    TASHKEEL_TTS_VOICE  صوت النطق الافتراضي، افتراضي ar-EG-ShakirNeural
    TTS_ENABLED      true|false يعطل/يفعل /tts ، افتراضي true
    TTS_TIMEOUT      مهلة التوليف بالثواني، افتراضي 45
    EDGE_TTS_VOICE   صوت السقوط الآلي لمحرك Edge (تُنقل إليه أصوات Gemini وتحل محلها)، افتراضي ar-EG-ShakirNeural
    EDGE_PROSOBY_STYLE تفعيل/تعطيل القِدر الأبطأ (-12%) ودرجة الصوت عند نطق Edge (افتراضي true)
    AZURE_TTS_KEY      مفتاح Azure Speech — F0 500K حرف/شهر مجاني دائم (سيرفر-محض)
    AZURE_REGION       منطقة Azure (مثال: eastus) — تُبنى منه الترويسة والرابط
    AZURE_TTS_VOICE    صوت Azure، افتراضي ar-EG-ShakirNeural (ذكور)
    AZURE_MAX_CHARS    حدّ أقصى لنص Azure لكل طلب، افتراضي 4000
    AZURE_TIMEOUT      مهلة استدعاء Azure بالثواني، افتراضي 40
    AZURE_COOLDOWN     كولداون السقوط بعد فشل Azure (ثوانٍ)، افتراضي 900
    CARTESIA_API_KEY   مفتاح Cartesia Sonic — 20K حرف/شهر مجاني متجدد (سيرفر-محض)
    CARTESIA_VOICE_ID  معرّف صوت عربي ذكور (يُجلب تلقائياً من القائمة إن تُرك فارغاً)
    CARTESIA_MAX_CHARS حدّ أقصى لنص Cartesia لكل طلب، افتراضي 2000
    CARTESIA_TIMEOUT   مهلة Cartesia بالثواني، افتراضي 30
    CARTESIA_COOLDOWN  كولداون السقوط بعد فشل Cartesia (ثوانٍ)، افتراضي 900
    TTS_ENGINE         auto (سلم Cartesia→Azure→Edge) | cartesia | azure | edge
    GEMINI_AUDIO_MODEL  نموذج الصوت الذكي، افتراضي gemini-2.5-flash-preview-tts (نموذج TTS مخصص — الصوت الأصلي غير مدعوم على النماذج العامة)
    GEMINI_AUDIO_MODELS قائمة سقوط مفصولة بفواصل؛ الأول ثم البدائل، افتراضي gemini-2.5-flash-preview-tts,gemini-3.1-flash-tts-preview,gemini-2.5-pro-preview-tts
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

# ── محركات الصوت الإضافية (حصص مجانية متجددة شهرية) ─────────────────────────
# Azure Speech F0 (500K حرف/شهر — دائم) + Cartesia Sonic 3.6 (20K حرف/شهر).
# المفاتيح سيرفر-محض عبر envs Render فقط؛ لا تُرسل للموّبايل ولا تُسجَّل.
AZURE_TTS_KEY = os.environ.get("AZURE_TTS_KEY", "").strip()
AZURE_REGION = os.environ.get("AZURE_REGION", "").strip().lower()
AZURE_TTS_VOICE = os.environ.get("AZURE_TTS_VOICE", "ar-EG-ShakirNeural").strip()
AZURE_MAX_CHARS = int(float(os.environ.get("AZURE_MAX_CHARS", "4000")))
AZURE_TIMEOUT = float(os.environ.get("AZURE_TIMEOUT", "40"))
AZURE_COOLDOWN = float(os.environ.get("AZURE_COOLDOWN", "900"))

CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "").strip()
CARTESIA_VOICE_ID = os.environ.get("CARTESIA_VOICE_ID", "").strip()
CARTESIA_MAX_CHARS = int(float(os.environ.get("CARTESIA_MAX_CHARS", "2000")))
CARTESIA_TIMEOUT = float(os.environ.get("CARTESIA_TIMEOUT", "30"))
CARTESIA_COOLDOWN = float(os.environ.get("CARTESIA_COOLDOWN", "900"))

# TTS_ENGINE: auto (السلم الكامل cartesia→azure→edge) | cartesia | azure | edge.
TTS_ENGINE = os.environ.get("TTS_ENGINE", "auto").strip().lower()

# حصص الشهر (نسبة التوقف الناعم 95% — عدّاد احترازي للسيرفر لا عدّاد فاتورة).
_ENGINE_QUOTAS = {"cartesia": 20000, "azure": 500000}
_ENGINE_SOFT_STOP_RATIO = 0.95

# ── CATT: محرك تشكيل محلي (Char-BERT ONNX) — Apache-2.0، بلا طلبات خارجية ──────
# الحزمة مدمجة (vendored) تحت catt_tashkeel/ ليبقى onnxruntime CPU فقط على Render
# (الحزمة الأصلية تُجبر onnxruntime-gpu/CUDA). النموذج EO (~74MB ONNX) يُنزَّل
# مرة واحدة من GitHub Releases ثم يُخزَّن محلياً؛ سقف النموذج 1024 حرفاً.
CATT_MAX_CHARS = int(float(os.environ.get("CATT_MAX_CHARS", "1024")))
CATT_TIMEOUT = float(os.environ.get("CATT_TIMEOUT", "60"))
# حدّ أدنى لاعتبار نتيجة CATT مشكَّلة (مثلاً النص الخام القصير يبقى بلا تنوين ظاهر).
CATT_MIN_LEN = int(float(os.environ.get("CATT_MIN_LEN", "0")))

# ── برادع الأسرار من السجلات (أمان المفاتيح) ─────────────────────────────────
class _SecretRedactor(logging.Filter):
    """يحجب أي شكل مفتاح قبل كتابة السجل: لا تظهر المفاتيح في أي مكان."""

    _PATS = (
        re.compile(r"sk_car_[A-Za-z0-9_.\-]+", re.I),
        re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
        re.compile(r"x-goog-api-key\s*[:=]\s*\S+", re.I),
        re.compile(r"ocp-apim-subscription-key\s*[:=]\s*\S+", re.I),
        re.compile(r"xi-api-key\s*[:=]\s*\S+", re.I),
        re.compile(r"authorization\s*[:=]\s*bearer\s+\S+", re.I),
    )

    def filter(self, record):
        try:
            msg = record.getMessage()
            for pat in self._PATS:
                msg = pat.sub("[REDACTED]", msg)
            record.msg = msg
            record.args = ()
        except Exception:  # noqa: BLE001 — البرادع لا يكسر التسجيل أبداً.
            pass
        return True


logger.addFilter(_SecretRedactor())

# ── حالات المحركات: كولداون سلبي + عداد حصص شهري (خيط آمن) ───────────────────
_engine_down_until = {}
_engine_down_lock = threading.Lock()
_engine_budget = {}
_engine_budget_lock = threading.Lock()


def _engine_blocked(name):
    """يعيد True إن كان المحرك في كولداون سلبي بعد فشل."""
    with _engine_down_lock:
        until = _engine_down_until.get(name, 0.0)
        if until > time.monotonic():
            return True
        if until:
            _engine_down_until.pop(name, None)
    return False


def _block_engine(name, reason):
    """يُدخل المحرك كولداوناً سلبياً (لا تُهدر المهل على محرك ميت)."""
    cooldown = CARTESIA_COOLDOWN if name == "cartesia" else AZURE_COOLDOWN
    with _engine_down_lock:
        _engine_down_until[name] = time.monotonic() + cooldown
    logger.warning("محرك %s متعذّر (%s)؛ يُتخطى لـ %ds.", name, reason, int(cooldown))


def _engine_budget_state(name):
    """(مستعمل، حصة، نسبة، شهر) — يُصفّر عداد الشهر تلقائياً عند تجدده."""
    month = time.strftime("%Y-%m")
    quota = _ENGINE_QUOTAS.get(name)
    if quota is None:
        return 0, None, 0.0, month
    with _engine_budget_lock:
        rec = _engine_budget.get(name)
        if not rec or rec["month"] != month:
            rec = {"month": month, "chars": 0}
            _engine_budget[name] = rec
        used = rec["chars"]
    return used, quota, used / quota, month


def _engine_soft_stopped(name):
    """التوقف الناعم: المحرك يُقصى لباقي الشهر عند 95% من حصته."""
    _, quota, pct, _ = _engine_budget_state(name)
    return quota is not None and pct >= _ENGINE_SOFT_STOP_RATIO


def _engine_charge(name, chars):
    """يحسب حروفاً مستهلكة من حصة المحرك (ناجحة فقط)."""
    with _engine_budget_lock:
        rec = _engine_budget.get(name)
        if rec and rec["month"] == time.strftime("%Y-%m"):
            rec["chars"] += max(0, int(chars))


_TASHKEEL_MARKS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)


def _strip_tashkeel(text):
    """يجرد علامات التشكيل لخفض الحروف المفوترة لدى المحركات ذات الحصص (15–30%)."""
    return _TASHKEEL_MARKS_RE.sub("", text or "")

# ── المحرك الذكي: صوت Gemini الأصلي (responseModalities AUDIO) ──
# الصوت الأصلي مدعوم على نماذج TTS المخصصة فقُط (النماذج العامة كـ gemini-3.6-flash
# تعيد نصاً لا صوتاً). GEMINI_AUDIO_MODEL يحدد الأول، وقائمة GEMINI_AUDIO_MODELS
# قائمة سقوط احتياطية تُجرَّب بالترتيب عند فشل/حجب سابق.
GEMINI_AUDIO_MODEL = (
    os.environ.get("GEMINI_AUDIO_MODEL", "gemini-2.5-flash-preview-tts")
    .strip()
    .replace("models/", "")
)
GEMINI_AUDIO_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_AUDIO_MODELS",
        "gemini-2.5-flash-preview-tts,gemini-3.1-flash-tts-preview,"
        "gemini-2.5-pro-preview-tts",
    ).split(",")
    if m.strip()
] or [GEMINI_AUDIO_MODEL]
if GEMINI_AUDIO_MODEL not in GEMINI_AUDIO_MODELS:
    GEMINI_AUDIO_MODELS.insert(0, GEMINI_AUDIO_MODEL)
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
#  CATT — تشكيل محلي دقيق (Char-BERT ONNX، Apache-2.0)
# ════════════════════════════════════════════════════════════════

_catt_model = None
_catt_lock = threading.Lock()
_catt_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="catt")

# مقاطع عربية فقط: حروف عربية + مسافات داخلية (المقطع=كلمة أو مجموعة كلمات).
# يقسّم النص بحيث يمرّ عبر CATT الجزءُ العربي وحده، ويُتركُ الترقيم/اللاتينية/
# الأرقام في أماكنها كما هي (do_tashkeel يمسح غير العربي لكننا نمرّره مقاطع نقية).
# [ \t]+ بين الكلمات فقط (لا مسافة نهاية) حتى لا تبتلع المسافة قبل الرقم/الترقيم.
_ARABIC_RUN_RE = re.compile(
    r"[\u0621-\u063A\u0640-\u064A]+(?:[ \t]+[\u0621-\u063A\u0640-\u064A]+)*"
)


def _get_catt_model():
    """يحمّل نموذج CATT EO مرة واحدة فقط (خيط آمن، تحميل كسول في أول استدعاء)."""
    global _catt_model
    if _catt_model is None:
        with _catt_lock:
            if _catt_model is None:
                from catt_tashkeel import CATTEncoderOnly  # حزمة مدمجة محلياً
                _catt_model = CATTEncoderOnly()
                logger.info("تم تحميل CATT EO (ONNX، CPU).")
    return _catt_model


def _catt_chunks(segment):
    """يقسّم مقطعاً عربياً (قد يعلو عن حد النموذج) إلى قطع ≤ CATT_MAX_CHARS على حدود الكلمات."""
    segment = segment.strip()
    if len(segment) <= CATT_MAX_CHARS:
        return [segment]
    out = []
    buf = []
    buf_len = 0
    for word in segment.split():
        w = len(word)
        if buf_len + (1 if buf else 0) + w > CATT_MAX_CHARS:
            if buf:
                out.append(" ".join(buf))
            buf = [word]
            buf_len = w
        else:
            buf.append(word)
            buf_len += (1 if len(buf) > 1 else 0) + w
    if buf:
        out.append(" ".join(buf))
    return out


def _catt_diacritize(text):
    """تشكيل محلي عبر CATT EO مع الحفاظ التام على الترقيم/الأرقام/اللاتينية.

    يُقسّم النص إلى مقاطع عربية نقية، يُشكَّل كلُّ مقطع (مع تقطيع ≤1024 حرفاً)،
    ثم يُعاد تركيبُ النص بحيث تبقى الرموز غير العربية في مواقعها. عند أي فشل
    تُعاد بقيةُ المقاطع كما هي — يُكمل السقوط الآلي.
    """
    model = _get_catt_model()
    parts = []
    pos = 0
    for m in _ARABIC_RUN_RE.finditer(text):
        seg = m.group(0)
        if not seg or not is_arabic_text(seg):
            continue
        parts.append(text[pos : m.start()])          # غير العربي قبل المقطع
        chunks = _catt_chunks(seg)
        try:
            out = model.do_tashkeel_batch(chunks, verbose=False)
            parts.append("".join(out))
        except Exception:
            logger.warning("CATT فشل على مقطع؛ يُكمل السقوط الآلي:\n%s",
                           traceback.format_exc())
            parts.append(seg)
        pos = m.end()
    parts.append(text[pos:])                          # ذيل قافي (غير عربي)
    return "".join(parts)


def _catt_diacritize_safe(text):
    """مسار CATT بمهلة زمنية صارمة؛ أي فشل/مهلة → None (فيُفعَّل ما بعده)."""
    if not text or not is_arabic_text(text):
        return None
    if len(text) > CATT_MAX_CHARS * 8:
        logger.info("نص طويل (%d حرفاً)؛ نتخطى CATT إلى المحرك التالي.", len(text))
        return None
    try:
        fut = _catt_pool.submit(_catt_diacritize, text)
        out = fut.result(timeout=CATT_TIMEOUT)
    except TimeoutError:
        logger.warning("CATT تجاوز المهلة (%ss)؛ ننتقل إلى البديل.", CATT_TIMEOUT)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("استدعاء CATT فشل: %s:%s", type(exc).__name__, exc)
        return None
    if not out or out == text:
        return None
    marks = len(_looks_diacritized_re.findall(out))
    if marks < CATT_MIN_LEN:
        return None
    return out


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
        f"{GEMINI_MODEL}:generateContent"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
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
    """سلسلة السقوط الآلي: Gemini→CATT→ONNX→النص الخام. لا تعود None أبداً.

    نصٌّ شُكِّل سابقاً (مثل ما يرسله التطبيق بعد تشكيل CATT/ONNX) يمرّ كما هو
    بمرحلة "pass" دون أي استدعاء — يحفظ الحصة المجانية.

    تُرجع (النص المٌشكَّل، اسم المرحلة المستخدمة: llm|catt|onnx|pass|raw).
    """
    if _looks_diacritized(text):
        return text, "pass"
    llm_out = _llm_diacritize_safe(text)
    if llm_out:
        return llm_out, "llm"
    catt_out = _catt_diacritize_safe(text)
    if catt_out:
        return catt_out, "catt"
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

# كاش سلبي لما يفشل صوت Gemini (404/غير مدعوم/مهلة/استجابة نصية): كلُّ نموذج
# يُتجاهل مؤقتاً حتى لا نُهدر محاولات في كل جملة. المفتاح = اسم النموذج.
_audio_neg_cache = {}
_audio_neg_lock = threading.Lock()
_AUDIO_NEG_COOLDOWN = float(os.environ.get("GEMINI_AUDIO_COOLDOWN", "900"))


def _audio_model_blocked(model):
    """بعد انتهاء كولداون العودة محاولة النموذج الصوتي. يعيد True إن كان محظوراً."""
    with _audio_neg_lock:
        until = _audio_neg_cache.get(model, 0.0)
        if until > time.monotonic():
            return True
        if until:
            _audio_neg_cache.pop(model, None)
        return False


def _block_audio_model(model, reason):
    """يسجّل فشل نموذج صوتي ويمنعه مؤقتاً في الطلبات التالية."""
    with _audio_neg_lock:
        _audio_neg_cache[model] = time.monotonic() + _AUDIO_NEG_COOLDOWN
    logger.warning(
        "نموذج الصوت %s يتعذّر (%s)؛ يُتخطى محاولاته لـ %is…",
        model,
        reason,
        int(_AUDIO_NEG_COOLDOWN),
    )


# أجزاء قد تحمل الصوت: Gemini TTS قد يعيد تعدد أجزاء (موجة صوتية واحدة عادةً).
def _extract_audio_from_parts(parts):
    """يبحث في كل الأجزاء عن أول inlineData صوتي. يعيد (bytes, mime) أو None."""
    for part in parts or []:
        inline = part.get("inlineData", {})
        mime = inline.get("mimeType", "")
        audio = base64.b64decode(inline.get("data", ""))
        if audio and len(audio) >= 100 and mime.startswith("audio/"):
            return audio, mime
    return None


def _gemini_audio(text, voice_name):
    """يولّد صوتاً أَمثل من Gemini عبر responseModalities:["AUDIO"].

    يعيد (bytes, mimetype) عند النجاح وإلا None (ينسَكب الواجهة إلى Edge).
    يجرب النماذج في GEMINI_AUDIO_MODELS بالترتيب، متخطياً المحظور مؤقتاً —
    والصوت الأصلي مدعوم على نماذج TTS المخصصة فقط.
    """
    if not GEMINI_API_KEY:
        return None
    for model in GEMINI_AUDIO_MODELS:
        if _audio_model_blocked(model):
            continue
        audio_mime = _gemini_audio_single(text, voice_name, model)
        if audio_mime:
            return audio_mime
    return None


def _gemini_audio_single(text, voice_name, model):
    """محاولة واحدة على نموذج صوتي محدد. يعيد (bytes, mime) أو None.

    عند رفض/فشل النموذج يُحظَر مؤقتاً (لكي يُجرَّب البديل في GEMINI_AUDIO_MODELS).
    """
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
        f"{model}:generateContent"
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=GEMINI_AUDIO_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error(
            "Gemini-Audio رفض الطلب على %s (%s) بتفاصيل: %s",
            model, exc.code, detail[:1200],
        )
        if exc.code in (400, 403, 404, 429):
            _block_audio_model(model, f"HTTP {exc.code}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("خطأ اتصال بـ Gemini-Audio (%s): %s:%s", model,
                     type(exc).__name__, exc)
        return None
    try:
        parts = data["candidates"][0]["content"].get("parts") or []
        found = _extract_audio_from_parts(parts)
        if found:
            return found
        # استجابة بلا صوت (نموذج نصي عام أو صوت محجوب): نوثّق ونحظر ثم نجرب البديل.
        texts = [p.get("text", "") for p in parts if p.get("text")]
        logger.warning(
            "Gemini-Audio (%s) استجابة بدون صوت؛ أول فينبك: %r — نص إذا وُجد: %s",
            model,
            json.dumps(data.get("promptFeedback", {}), ensure_ascii=False)[:200],
            (" ".join(texts))[:120] or "(بلا نص)",
        )
        _block_audio_model(model, "استجابة نصية/صامتة")
        return None
    except (KeyError, IndexError, TypeError):
        logger.error("استجابة Gemini-Audio بلا inlineData (%s): %s",
                     model, json.dumps(data, ensure_ascii=False)[:400])
        _block_audio_model(model, "بلا candidates")
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


def _azure_enabled():
    """Azure محرّك متاح فقط إن وُجد المفتاح والمنطقة (سيرفر-محض) مع اسم منطقة آمن."""
    return bool(AZURE_TTS_KEY and AZURE_REGION) and (
        re.fullmatch(r"[a-z0-9-]+", AZURE_REGION) is not None
    )


def _azure_ssml(text, voice, rate_pct):
    """SSML خفيف لـ Azure: وقفات الفواصل/الأسطر، وprosody عند معدل غير صفري فقط."""
    inner = _ssml_inner(text)
    if rate_pct:
        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xml:lang="ar-SA"><voice name="{voice}">'
            f'<prosody rate="{rate_pct:+d}%">{inner}</prosody>'
            f"</voice></speak>"
        )
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="ar-SA"><voice name="{voice}">{inner}</voice></speak>'
    )


def _azure_tts_bytes(text, rate_pct):
    """يولّف عبر Azure Speech REST (بلا SDK ثقيل) ويعيد MP3 bytes أو None.

    أي خطأ يُدخل المحرك كولداوناً سلبياً (لا تُهدر مهل ناجية مع محرك ميت).
    العداد الشهري يُشحن فقط بعد نجاح فعلي (F0 لا يُفوَّر أبداً).
    """
    if not _azure_enabled() or _engine_soft_stopped("azure") or _engine_blocked("azure"):
        return None
    if len(text) > AZURE_MAX_CHARS:
        logger.info("نص فوق %d حرفاً؛ نتجاوز Azure إلى Edge.", AZURE_MAX_CHARS)
        return None
    voice = AZURE_TTS_VOICE
    ssml = _azure_ssml(text, voice, rate_pct)
    req = urllib.request.Request(
        f"https://{AZURE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1",
        data=ssml.encode("utf-8"),
        headers={
            "Ocp-Apim-Subscription-Key": AZURE_TTS_KEY,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=AZURE_TIMEOUT) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error("Azure رفض الطلب (HTTP %s) — %s", exc.code, detail[:600])
        if exc.code in (400, 401, 403, 429):
            _block_engine("azure", f"HTTP {exc.code}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("خطأ اتصال بـ Azure: %s:%s", type(exc).__name__, exc)
        _block_engine("azure", type(exc).__name__)
        return None
    if not _looks_like_mp3(audio):
        logger.error("Azure استجابة ليست MP3 صالحة: %d بايت.", len(audio))
        _block_engine("azure", "bad mp3")
        return None
    _engine_charge("azure", len(_strip_tashkeel(text)))
    return audio


_cartesia_voice_resolved = None
_cartesia_voice_lock = threading.Lock()


def _cartesia_voice_id():
    """يعيد معرّف صوت Cartesia: CARTESIA_VOICE_ID مباشرة، أو (مرة واحدة) جلب
    قائمة الأصوات العربية واختيار صوت ذكور. لا يُعاد الجلب إلا عند غياب المفتاح."""
    global _cartesia_voice_resolved
    if CARTESIA_VOICE_ID:
        return CARTESIA_VOICE_ID
    if _cartesia_voice_resolved:
        return _cartesia_voice_resolved
    if not CARTESIA_API_KEY:
        return None
    with _cartesia_voice_lock:
        if _cartesia_voice_resolved:
            return _cartesia_voice_resolved
        picked = None
        try:
            req = urllib.request.Request(
                "https://api.cartesia.ai/voices?language=ar",
                headers={
                    "X-API-Key": CARTESIA_API_KEY,
                    "Authorization": f"Bearer {CARTESIA_API_KEY}",
                },
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            voices = list((data or {}).get("voices") or [])
            for v in voices:
                gender = str(v.get("gender") or v.get("sex") or "").lower()
                if gender and gender != "female":
                    picked = v.get("id")
                    break
            if not picked:
                picked = voices[0].get("id") if voices else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("تعذّر جلب أصوات Cartesia (%s:%s)؛ يُتخطى المحرك.",
                           type(exc).__name__, exc)
        _cartesia_voice_resolved = picked
        if picked:
            logger.info("Cartesia: صوت عربي تلقائي %s.", picked)
        return picked


def _cartesia_tts_bytes(text, rate_pct):
    """يولّف عبر Cartesia Sonic (الاستجابة ~17ms) ويعيد MP3 bytes أو None.

    النص يُجرَّد من علامات التشكيل (حروف مُفوترة أقل). أي خطأ يُدخل كولداوناً.
    """
    if not CARTESIA_API_KEY or _engine_soft_stopped("cartesia") or _engine_blocked("cartesia"):
        return None
    voice_id = _cartesia_voice_id()
    if not voice_id:
        logger.info("لا يوجد صوت Cartesia معرّف؛ نتجاوزه إلى Azure.")
        return None
    if len(text) > CARTESIA_MAX_CHARS:
        logger.info("نص فوق %d حرفاً؛ نتجاوز Cartesia إلى Azure/Edge.", CARTESIA_MAX_CHARS)
        return None
    payload = {
        "model_id": "sonic-3.6",
        "transcript": text,
        "voice": {"id": voice_id},
        "language": "ar",
        "output_format": {"container": "mp3", "sample_rate": 24000, "bit_rate": 96000},
    }
    if rate_pct:
        speed = min(1.5, max(0.6, 1.0 + rate_pct / 100.0))
        payload["generation_config"] = {"speed": speed}
    req = urllib.request.Request(
        "https://api.cartesia.ai/tts/bytes",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "X-API-Key": CARTESIA_API_KEY,
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Cartesia-Version": "2026-08-14",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=CARTESIA_TIMEOUT) as resp:
            audio = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error("Cartesia رفض الطلب (HTTP %s) — %s", exc.code, detail[:600])
        if exc.code in (400, 401, 403, 404, 429):
            _block_engine("cartesia", f"HTTP {exc.code}")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("خطأ اتصال بـ Cartesia: %s:%s", type(exc).__name__, exc)
        _block_engine("cartesia", type(exc).__name__)
        return None
    if not _looks_like_mp3(audio):
        logger.error("Cartesia استجابة ليست MP3 صالحة: %d بايت.", len(audio))
        _block_engine("cartesia", "bad mp3")
        return None
    _engine_charge("cartesia", len(text))
    return audio


def _synth_ladder(text, voice, rate):
    """سلم المحركات: تشكيل واحد ← Cartesia→Azure→Edge (حسب TTS_ENGINE).

    يعيد (audio|None, engine: cartesia|azure|edge, voice_used, diac_engine).
    قواعد:
      • محركات الحصّة (Cartesia/Azure) تُفوَّر بنصٍّ مجرَّد من التشكيل
        (حروف مُفوترة أقل)؛ إعادة المُشكَّل حصراً للسقوط الأخير (Edge).
      • TTS_ENGINE يقيّد خيارات الحصّة فقط؛ Edge يبقى الملاذ الأخير دائماً
        (نطق مشكَّل ثم خام) — لا يتوقف مطلقاً، ويرفع أي صوت محترم بتفجيعات.
    """
    preferred = TTS_ENGINE if TTS_ENGINE != "auto" else None
    candidates = ["cartesia", "azure"] if preferred is None else ([preferred] if preferred in (
        "cartesia", "azure") else [])
    result, diac_engine = context_diacritize(text)
    stripped = _strip_tashkeel(result)

    for name in candidates:
        if name == "cartesia":
            audio = _cartesia_tts_bytes(stripped, rate)
            if audio is not None:
                return audio, "cartesia", _cartesia_voice_id() or "cartesia:auto", diac_engine
        else:
            audio = _azure_tts_bytes(stripped, rate)
            if audio is not None:
                return audio, "azure", AZURE_TTS_VOICE, diac_engine

    voice_used = _edge_voice(voice)
    audio = synthesize(result, voice_used, rate)
    engine = diac_engine
    if audio is None:
        logger.warning("تأخر/فشل نطق النص المشكَّل عبر Edge؛ ننطق النص الخام.")
        audio = synthesize(text, voice_used, rate)
        engine = "raw"
    return audio, "edge", voice_used, engine


@app.route("/audio/tts", methods=["GET"])
def audio_tts_get():
    """نفس /tts لكن عبر GET — للبث المباشر لدى just_audio/ExoPlayer مع Range (206).

    السلم الكامل (TTS_ENGINE): Cartesia→Azure→Edge، بنفس انضباط /tts.
    الرؤوس: X-Tashkeel-Engine = cartesia|azure|edge ، X-Tashkeel-Diacrit ، X-Voice.
    """
    if not TTS_ENABLED:
        return jsonify({"error": "tts disabled"}), 404
    text = (request.args.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    voice = request.args.get("voice") or TTS_VOICE
    rate = _parse_rate(request.args.get("rate", "0"))
    audio, engine, voice_used, diac_engine = _synth_ladder(text, voice, rate)
    if audio is None:
        return jsonify({"error": "tts synthesis failed"}), 500
    resp = _audio_response(audio, "audio/mpeg")
    resp.headers["X-Tashkeel-Engine"] = engine
    resp.headers["X-Tashkeel-Diacrit"] = diac_engine
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
    """العقد الإلزامي القديم — توافق كامل مع تطبيق الموبايل، بدقة أعلى الآن.

    يستخدم السلم السياقي (Gemini→CATT→ONNX→الخام) ليرتفع جودةُ النص الذي
    يعرضه التطبيق ويعيد إرساله إلى النطق. إضافةً للسابق «engine» اختيارية
    (لا يكسر تطبيقاً يقرأ «diacritized» فقط).
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    if not is_arabic_text(text):
        # نص بلا حروف عربية: يُعاد كما هو دون تشكيل (مكافئ FR-4)
        return jsonify({"diacritized": text, "engine": "pass"})
    try:
        result, engine = context_diacritize(text)
        out = {"diacritized": result}
        if engine:
            out["engine"] = engine
        return jsonify(out)
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
    """خط أنابيب كامل: تشكيل سياقي → سلم المحركات Cartesia→Azure→Edge.

    السقوط الآلي: أي محرك حصّة فاشل يُتخطى؛ وEdge ناطق النص المشكَّل ثم الخام —
    فلا يتوقف التطبيق.
    الرؤوس: X-Tashkeel-Engine = cartesia|azure|edge ، X-Tashkeel-Diacrit = llm|catt|onnx|raw ،
    X-Voice المستخدم الفعلي.
    """
    if not TTS_ENABLED:
        return jsonify({"error": "tts disabled"}), 404
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty text"}), 400
    voice = data.get("voice") or TTS_VOICE
    rate = _parse_rate(data.get("rate", "0"))

    audio, engine, voice_used, diac_engine = _synth_ladder(text, voice, rate)
    if audio is None:
        return jsonify({"error": "tts synthesis failed"}), 500

    return Response(
        audio,
        mimetype="audio/mpeg",
        headers={
            "X-Tashkeel-Engine": engine,
            "X-Tashkeel-Diacrit": diac_engine,
            "X-Voice": voice_used,
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


@app.route("/tts/status", methods=["GET"])
def tts_status():
    """تشخيص سريع: أي محرك مُفعّل/مُستنفد/في كولداون + الحصص الشهرية.

    أرقام ومنطق فقط — لا يُعرض أي مفتاح ولا سر.
    """
    def state(name):
        used, quota, pct, month = _engine_budget_state(name)
        configured = (name == "cartesia" and bool(CARTESIA_API_KEY)) or (
            name == "azure" and _azure_enabled()
        )
        return {
            "configured": configured,
            "active": configured
            and not _engine_soft_stopped(name)
            and not _engine_blocked(name),
            "cooldown": _engine_blocked(name),
            "soft_stopped": _engine_soft_stopped(name),
            "usage_pct": round(pct * 100, 1),
            "chars_used": used,
            "chars_quota": quota,
            "quota_month": month,
        }

    if TTS_ENGINE == "auto":
        effective = ["cartesia", "azure", "edge"]
    elif TTS_ENGINE in ("cartesia", "azure"):
        effective = [TTS_ENGINE, "edge"]
    else:
        effective = ["edge"]

    return jsonify({
        "tts_enabled": TTS_ENABLED,
        "engine_mode": TTS_ENGINE,
        "effective_order": effective,
        "engines": {"cartesia": state("cartesia"), "azure": state("azure")},
        "edge": {"enabled": TTS_ENABLED, "voice": EDGE_FALLBACK_VOICE},
    })


# شحن كاش LLM من القرص عند بدء التشغيل (لا يُهدر الحصة على النصوص المتكررة).
_load_disk_cache()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)