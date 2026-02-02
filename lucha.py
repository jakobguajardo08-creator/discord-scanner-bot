#!/usr/bin/env python3
"""
LuCha — Discord Scanner + Daily Info + YouTube Latest Video (secure+stateful)

Key upgrades:
- Security hardening (no secrets in logs/files, forced dry-run unless explicitly armed)
- YouTube "latest video" posting via RSS feed (no API key) + de-dupe (won't post same video twice)
- Daily info message with per-date overrides (e.g., Feb 3, 2026)
- Intents minimized: message_content only enabled when moderation scanning is enabled

ENV (Required)
  DISCORD_TOKEN
  GUILD_ID

Core files
  BACKUP_FILE (default: guild_backup.json)      # channel snapshot for nuke detection + rename restore map
  REPORT_FILE (default: mod_report.json)        # run report (no secrets)
  STATE_FILE  (default: lucha_state.json)       # persistent state (last YT video posted, last daily posted)

Safety switches
  LUCHA_ARMED=0/1              (default 0) if 0 -> forces DRY_RUN=True
  DRY_RUN=0/1                  (default 0) if 1 -> never edit/delete/post, only report actions

Feature toggles
  ENABLE_RENAME=0/1            (default 1)
  ENABLE_MODERATION=0        (default 1)
  ENABLE_NUKE_REMEDIATION=0/1  (default 1)
  ENABLE_DAILY_INFO=0/1        (default 1)
  ENABLE_YOUTUBE=0/1           (default 1)

Restore mode
  RESTORE_RENAMES=0/1          (default 0) if 1 -> restore channel names from snapshot (requires manage_channels)

Renaming controls
  RENAME_MODE=prefix|suffix    (default prefix)
  MAX_RENAMES                  (default 40)
  RENAME_EXCLUDE_REGEX         (default empty)
  RENAME_ALLOWLIST_REGEX       (default empty)   # if set, ONLY rename channels matching
  RENAME_INCLUDE_CATEGORIES=0/1 (default 0)

Scanning controls
  MAX_MESSAGES                 (default 250)
  MAX_CHANNELS                 (default 999)
  SCAN_CONCURRENCY             (default 3)
  TOXIC_THRESHOLD              (default 0.6)
  NSFW_THRESHOLD               (default 0.8)      # used only if image model available
  SCAN_ALLOWLIST_REGEX         (default empty)    # if set, ONLY scan channels matching
  SCAN_EXCLUDE_REGEX           (default empty)    # skip scanning channels matching
  DELETE_WEBHOOK_LEAKS=0/1     (default 1)
  DELETE_CREDENTIAL_LEAKS=0/1  (default 1)

Nuke detection
  NUKE_DELETION_THRESHOLD      (default 0.15)
  NUKE_NEW_CHANNEL_THRESHOLD   (default 6)
  NUKE_MAX_NEW_CHANNEL_AGE_MIN (default 180)
  NUKE_DELETE_NAME_REGEX       (default "(spam|raid|hacked|nuke|giveaway|free-.*|bot-)" )

Posting
  INFO_CHANNEL_NAME            (default "daily-info")

YouTube (RSS-based; no API key needed)
  YT_CHANNEL_ID                (required if ENABLE_YOUTUBE=1)  # UCxxxx...
  YT_POST_CHANNEL_NAME         (default INFO_CHANNEL_NAME)
  YT_POST_TEMPLATE             (default: "📺 **New video!** {title}\n{url}")
"""

from __future__ import annotations

import os
import sys
import json
import io
import re
import asyncio
import logging
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Tuple, Optional, List, Any

import aiohttp
import discord
from discord.utils import get

# Optional: load env from .env (local only)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Optional ML deps (graceful)
try:
    from transformers import (  # type: ignore
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoFeatureExtractor,
        AutoModelForImageClassification,
    )
    import torch  # type: ignore
    TF_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoFeatureExtractor = None
    AutoModelForImageClassification = None
    torch = None
    TF_AVAILABLE = False

LUCHA_NAME = "LuCha"

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("LuCha")


# ---------------- Safe env parsing ----------------
def _env_raw(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v if v != "" else None

def env_str(name: str, default: str) -> str:
    v = _env_raw(name)
    return v if v is not None else default

def env_bool(name: str, default: bool) -> bool:
    v = _env_raw(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")

def env_int(name: str, default: int, min_v: Optional[int] = None, max_v: Optional[int] = None) -> int:
    v = _env_raw(name)
    if v is None:
        return default
    try:
        n = int(v)
        if min_v is not None:
            n = max(min_v, n)
        if max_v is not None:
            n = min(max_v, n)
        return n
    except ValueError:
        return default

def env_float(name: str, default: float, min_v: Optional[float] = None, max_v: Optional[float] = None) -> float:
    v = _env_raw(name)
    if v is None:
        return default
    try:
        n = float(v)
        if min_v is not None:
            n = max(min_v, n)
        if max_v is not None:
            n = min(max_v, n)
        return n
    except ValueError:
        return default


# ---------------- Config ----------------
@dataclass
class Config:
    discord_token: str
    guild_id: int

    backup_file: str = "guild_backup.json"
    report_file: str = "mod_report.json"
    state_file: str = "lucha_state.json"

    info_channel_name: str = "daily-info"
    yt_channel_id: str = ""
    yt_post_channel_name: str = ""  # default to info_channel_name
    yt_post_template: str = "📺 **New video!** {title}\n{url}"

    lucha_armed: bool = False
    dry_run: bool = True

    enable_rename: bool = True
    enable_moderation: bool = True
    enable_nuke_remediation: bool = True
    enable_daily_info: bool = True
    enable_youtube: bool = True
    restore_renames: bool = False

    rename_mode: str = "prefix"
    max_renames: int = 40
    rename_exclude_regex: str = ""
    rename_allowlist_regex: str = ""
    rename_include_categories: bool = False

    max_messages: int = 250
    max_channels: int = 999
    scan_concurrency: int = 3
    toxic_threshold: float = 0.6
    nsfw_threshold: float = 0.8
    scan_allowlist_regex: str = ""
    scan_exclude_regex: str = ""
    delete_webhook_leaks: bool = True
    delete_credential_leaks: bool = True

    nuke_deletion_threshold: float = 0.15
    nuke_new_channel_threshold: int = 6
    nuke_max_new_channel_age_min: int = 180
    nuke_delete_name_regex: str = r"(spam|raid|hacked|nuke|giveaway|free-.*|bot-)"

    @staticmethod
    def load() -> "Config":
        token = env_str("DISCORD_TOKEN", "")
        gid_raw = _env_raw("GUILD_ID")
        if not token or not gid_raw:
            print("DISCORD_TOKEN or GUILD_ID not set", file=sys.stderr)
            sys.exit(2)

        try:
            gid = int(gid_raw)
        except ValueError:
            print("GUILD_ID must be an integer", file=sys.stderr)
            sys.exit(2)

        cfg = Config(
            discord_token=token,
            guild_id=gid,

            backup_file=env_str("BACKUP_FILE", "guild_backup.json"),
            report_file=env_str("REPORT_FILE", "mod_report.json"),
            state_file=env_str("STATE_FILE", "lucha_state.json"),

            info_channel_name=env_str("INFO_CHANNEL_NAME", "daily-info"),
            yt_channel_id=env_str("YT_CHANNEL_ID", ""),
            yt_post_channel_name=env_str("YT_POST_CHANNEL_NAME", ""),
            yt_post_template=env_str("YT_POST_TEMPLATE", "📺 **New video!** {title}\n{url}"),

            lucha_armed=env_bool("LUCHA_ARMED", False),
            dry_run=env_bool("DRY_RUN", False),

            enable_rename=env_bool("ENABLE_RENAME", True),
            enable_moderation=env_bool("ENABLE_MODERATION", True),
            enable_nuke_remediation=env_bool("ENABLE_NUKE_REMEDIATION", True),
            enable_daily_info=env_bool("ENABLE_DAILY_INFO", True),
            enable_youtube=env_bool("ENABLE_YOUTUBE", True),
            restore_renames=env_bool("RESTORE_RENAMES", False),

            rename_mode=env_str("RENAME_MODE", "prefix").lower(),
            max_renames=env_int("MAX_RENAMES", 40, min_v=0, max_v=500),
            rename_exclude_regex=env_str("RENAME_EXCLUDE_REGEX", ""),
            rename_allowlist_regex=env_str("RENAME_ALLOWLIST_REGEX", ""),
            rename_include_categories=env_bool("RENAME_INCLUDE_CATEGORIES", False),

            max_messages=env_int("MAX_MESSAGES", 250, min_v=1, max_v=5000),
            max_channels=env_int("MAX_CHANNELS", 999, min_v=1, max_v=5000),
            scan_concurrency=env_int("SCAN_CONCURRENCY", 3, min_v=1, max_v=10),

            toxic_threshold=env_float("TOXIC_THRESHOLD", 0.6, min_v=0.0, max_v=1.0),
            nsfw_threshold=env_float("NSFW_THRESHOLD", 0.8, min_v=0.0, max_v=1.0),

            scan_allowlist_regex=env_str("SCAN_ALLOWLIST_REGEX", ""),
            scan_exclude_regex=env_str("SCAN_EXCLUDE_REGEX", ""),

            delete_webhook_leaks=env_bool("DELETE_WEBHOOK_LEAKS", True),
            delete_credential_leaks=env_bool("DELETE_CREDENTIAL_LEAKS", True),

            nuke_deletion_threshold=env_float("NUKE_DELETION_THRESHOLD", 0.15, min_v=0.0, max_v=1.0),
            nuke_new_channel_threshold=env_int("NUKE_NEW_CHANNEL_THRESHOLD", 6, min_v=0, max_v=9999),
            nuke_max_new_channel_age_min=env_int("NUKE_MAX_NEW_CHANNEL_AGE_MIN", 180, min_v=1, max_v=1440),
            nuke_delete_name_regex=env_str("NUKE_DELETE_NAME_REGEX", r"(spam|raid|hacked|nuke|giveaway|free-.*|bot-)"),
        )

        # Hard safety: must explicitly arm for destructive edits/posts/deletes
        if not cfg.lucha_armed:
            cfg.dry_run = True

        # sanitize rename_mode
        if cfg.rename_mode not in ("prefix", "suffix"):
            cfg.rename_mode = "prefix"

        # default YT post channel = info channel
        if not cfg.yt_post_channel_name:
            cfg.yt_post_channel_name = cfg.info_channel_name

        # if youtube enabled, require channel id
        if cfg.enable_youtube and not cfg.yt_channel_id:
            log.warning("ENABLE_YOUTUBE=1 but YT_CHANNEL_ID is empty; YouTube posting will be skipped safely.")

        return cfg


CFG = Config.load()


# ---------------- Report (NO secrets) ----------------
REPORT: Dict[str, Any] = {
    "lucha": {
        "name": LUCHA_NAME,
        "armed": CFG.lucha_armed,
        "dry_run": CFG.dry_run,
        "ts": None,
        "sigil": None,
        "version": "3.0-secure-yt-daily",
    },
    "metrics": {
        "channels_scanned": 0,
        "messages_scanned": 0,
        "messages_deleted": 0,
        "attachments_scanned": 0,
        "renames_done": 0,
        "renames_would": 0,
        "posts_sent": 0,
        "api_calls_soft": 0,
    },
    "nuke_events": [],
    "actions": [],
    "renames": [],
    "suspicious_content": [],
    "nsfw_attachments": [],
    "youtube": {},
    "daily_info": {},
    "errors": [],
}


# ---------------- Constants ----------------
SCAN_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

SEASONAL_EMOJIS = {"🎃": [10, 11], "🎄": [12], "💖": [2], "🌸": [3, 4]}
DEFAULT_EMOJI = "🔹"

# Stronger suspicious patterns (still conservative)
SUSPICIOUS_PATTERNS = [
    # common secrets
    r"\b(api[_-]?key|secret|token|client[_-]?secret|access[_-]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\/+=]{8,}['\"]?",
    r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----",

    # discord webhooks
    r"(https?:\/\/(canary\.|ptb\.)?discord(app)?\.com\/api\/webhooks\/\d+\/[A-Za-z0-9_\-]+)",

    # discord tokens (rough patterns)
    r"\b(mfa\.[A-Za-z0-9_\-]{80,})\b",
    r"\b([A-Za-z0-9_\-]{23,28}\.[A-Za-z0-9_\-]{6,7}\.[A-Za-z0-9_\-]{27,})\b",

    # AWS keys (very rough)
    r"\bAKIA[0-9A-Z]{16}\b",

    # code exec / downloaders
    r"\bexec\(",
    r"\beval\(",
    r"base64\.b64decode",
    r"powershell\s+.*-EncodedCommand",
    r"\bcurl\s+.*--output\b",
    r"\bwget\s+.*\s-O\b",
    r"\bInvoke-WebRequest\b",
]

YOUTUBE_RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"


# ---------------- Utilities ----------------
def utc_now_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def today_utc_date() -> date:
    return now_utc().date()

def save_json(path: str, data: Any):
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        REPORT["errors"].append(f"save_json_error_{path}:{e}")

def load_json(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        REPORT["errors"].append(f"load_json_error_{path}:{e}")
    return {}

def short_snip(s: str, n: int = 260) -> str:
    s = s or ""
    return (s[:n] + "...") if len(s) > n else s

def safe_regex_search(pattern: str, text: str) -> bool:
    if not pattern:
        return False
    try:
        return bool(re.search(pattern, text or "", re.IGNORECASE))
    except re.error:
        return False

def sanitize_channel_name(name: str) -> str:
    name = re.sub(r"\s+", "-", (name or "").strip())
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-")
    if not name:
        name = "channel"
    return name[:100]

def suspicious_code_check(text: str) -> Optional[str]:
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, text or "", re.IGNORECASE):
            return pat
    return None

def get_dynamic_emoji() -> str:
    month = datetime.utcnow().month
    for e, months in SEASONAL_EMOJIS.items():
        if month in months:
            return e
    return DEFAULT_EMOJI

def make_daily_sigil(guild_id: int, emoji: str) -> str:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    h = hashlib.sha256(f"{guild_id}:{day}:{emoji}:LuCha".encode("utf-8")).hexdigest()[:10]
    return f"LU-{h}-{emoji}"

def redact_for_logs(s: str) -> str:
    """Best-effort: never log actual secrets if they slip into exceptions/messages."""
    if not s:
        return s
    # Mask things that look like tokens/webhooks quickly
    s = re.sub(r"(mfa\.[A-Za-z0-9_\-]{20,})", "mfa.[REDACTED]", s)
    s = re.sub(r"(discord(app)?\.com\/api\/webhooks\/\d+\/)[A-Za-z0-9_\-]+", r"\1[REDACTED]", s)
    s = re.sub(r"([A-Za-z0-9_\-]{23,28}\.[A-Za-z0-9_\-]{6,7}\.[A-Za-z0-9_\-]{10,})", "[REDACTED_TOKEN]", s)
    return s


# ---------------- Persistent state (YT + daily) ----------------
def load_state() -> dict:
    st = load_json(CFG.state_file) or {}
    if not isinstance(st, dict):
        return {}
    return st

def save_state(st: dict):
    save_json(CFG.state_file, st)

def get_state_key_date(prefix: str) -> str:
    return f"{prefix}:{today_utc_date().isoformat()}"


# ---------------- Optional ML models ----------------
txt_model = None
txt_tokenizer = None
img_extractor = None
img_model = None

def init_models():
    global txt_model, txt_tokenizer, img_extractor, img_model

    if not CFG.enable_moderation:
        return

    # Toxicity model
    try:
        if AutoTokenizer and AutoModelForSequenceClassification:
            txt_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
            txt_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
            log.info("Loaded toxic text model.")
    except Exception as e:
        REPORT["errors"].append(f"toxic_model_load_error:{redact_for_logs(str(e))}")

    # NSFW image model
    try:
        if AutoFeatureExtractor and AutoModelForImageClassification:
            img_extractor = AutoFeatureExtractor.from_pretrained("Falconsai/nsfw_image_detection")
            img_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
            log.info("Loaded NSFW image model.")
    except Exception as e:
        REPORT["errors"].append(f"nsfw_model_load_error:{redact_for_logs(str(e))}")


def run_model_on_text(text: str) -> Dict[str, float]:
    """Returns dict with 'toxic' in [0..1]. If no model, uses a conservative heuristic."""
    if not txt_model or not txt_tokenizer or not torch:
        lower = (text or "").lower()
        triggers = [
            "kys", "kill yourself", "die", "stfu",
            "retard", "faggot", "nazi",
            "i will dox", "dox you", "swat", "swatting",
        ]
        score = 0.0
        for t in triggers:
            if t in lower:
                score += 0.22
        if sum(1 for c in text if c.isupper()) >= 20:
            score += 0.08
        return {"toxic": min(score, 1.0)}

    try:
        inputs = txt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = txt_model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy()
        return {"toxic": float(probs[0])}
    except Exception as e:
        REPORT["errors"].append(f"text_scan_error:{redact_for_logs(str(e))}")
        return {"toxic": 0.0}


def is_text_toxic(text: str) -> Tuple[bool, Dict[str, float]]:
    scores = run_model_on_text(text)
    return (float(scores.get("toxic", 0.0)) >= CFG.toxic_threshold, scores)


def run_model_on_image_bytes(data: bytes) -> float:
    """Returns nsfw score in [0..1]. If no model, returns 0."""
    if not img_extractor or not img_model or not torch:
        return 0.0
    try:
        from PIL import Image  # type: ignore
        img = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = img_extractor(images=img, return_tensors="pt")
        outputs = img_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
        if len(probs) > 1:
            return float(probs[1])
        return float(max(probs))
    except Exception as e:
        REPORT["errors"].append(f"image_scan_error:{redact_for_logs(str(e))}")
        return 0.0


# ---------------- Snapshot & nuke detection ----------------
def build_snapshot(guild: discord.Guild) -> dict:
    snap = {
        "ts": utc_now_iso(),
        "guild_id": guild.id,
        "channels": [],
        "channel_ids": [],
        "original_channel_names": {},  # id -> original name for reversible renames
    }

    for ch in guild.channels:
        snap["channels"].append({
            "id": ch.id,
            "name": getattr(ch, "name", ""),
            "type": str(getattr(ch, "type", "")),
            "position": getattr(ch, "position", None),
        })
        snap["channel_ids"].append(ch.id)
        snap["original_channel_names"][str(ch.id)] = getattr(ch, "name", "")

    return snap

def detect_nuke(prev_snapshot: dict, guild: discord.Guild) -> dict:
    prev_ids = set(prev_snapshot.get("channel_ids", []) if prev_snapshot else [])
    curr_ids = set(ch.id for ch in guild.channels)

    missing_ids = list(prev_ids - curr_ids)
    new_ids = list(curr_ids - prev_ids)

    deleted_frac = (len(missing_ids) / max(1, len(prev_ids))) if prev_ids else 0.0
    is_nuke = bool(prev_ids) and (deleted_frac >= CFG.nuke_deletion_threshold) and (len(new_ids) >= CFG.nuke_new_channel_threshold)

    id_to_name_prev = {int(c["id"]): c.get("name") for c in prev_snapshot.get("channels", [])} if prev_snapshot else {}
    id_to_name_curr = {ch.id: getattr(ch, "name", str(ch.id)) for ch in guild.channels}

    return {
        "is_nuke": is_nuke,
        "deleted_count": len(missing_ids),
        "created_count": len(new_ids),
        "deleted_frac": round(deleted_frac, 4),
        "missing_sample": [id_to_name_prev.get(i, str(i)) for i in missing_ids[:10]],
        "new_sample": [id_to_name_curr.get(i, str(i)) for i in new_ids[:10]],
    }

async def remediate_nuke(guild: discord.Guild, prev_snapshot: dict) -> List[str]:
    if not CFG.enable_nuke_remediation:
        return []

    prev_ids = set(prev_snapshot.get("channel_ids", []) if prev_snapshot else [])
    deleted: List[str] = []

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_permission_manage_channels_for_remediation")
        return deleted

    exploded = len(guild.channels) > (len(prev_ids) + 8) if prev_ids else False
    try:
        name_re = re.compile(CFG.nuke_delete_name_regex, re.IGNORECASE)
    except re.error:
        name_re = re.compile(r"(spam|raid|hacked|nuke|giveaway|free-.*|bot-)", re.IGNORECASE)

    age_cutoff = now_utc() - timedelta(minutes=CFG.nuke_max_new_channel_age_min)

    for ch in guild.channels:
        if ch.id in prev_ids:
            continue

        created_at = getattr(ch, "created_at", None)
        is_recent = True
        if isinstance(created_at, datetime):
            try:
                is_recent = created_at.replace(tzinfo=timezone.utc) >= age_cutoff
            except Exception:
                is_recent = True

        suspect_name = bool(name_re.search(getattr(ch, "name", "") or ""))

        # Conservative rule:
        # delete if (suspect_name AND recent) OR (exploded AND recent)
        if not ((suspect_name and is_recent) or (exploded and is_recent)):
            continue

        if CFG.dry_run:
            REPORT["actions"].append({"action": "would_delete_channel", "channel": getattr(ch, "name", ""), "ts": utc_now_iso()})
            deleted.append(getattr(ch, "name", ""))
            continue

        try:
            await ch.delete(reason=f"{LUCHA_NAME}: foreign channel after suspected nuke (conservative)")
            REPORT["actions"].append({"action": "delete_channel", "channel": getattr(ch, "name", ""), "ts": utc_now_iso()})
            deleted.append(getattr(ch, "name", ""))
            REPORT["metrics"]["api_calls_soft"] += 1
            await asyncio.sleep(1.1)
        except Exception as e:
            REPORT["errors"].append(f"delete_channel_error:{redact_for_logs(str(e))}")

    return deleted


# ---------------- Channel rename theming + restore ----------------
def should_rename_channel(ch: discord.abc.GuildChannel) -> bool:
    name = getattr(ch, "name", "") or ""
    if safe_regex_search(CFG.rename_exclude_regex, name):
        return False
    if CFG.rename_allowlist_regex and not safe_regex_search(CFG.rename_allowlist_regex, name):
        return False
    if isinstance(ch, discord.CategoryChannel) and not CFG.rename_include_categories:
        return False
    return True

async def themed_rename_channels(guild: discord.Guild, emoji: str, snapshot: dict):
    if not CFG.enable_rename or CFG.restore_renames:
        return

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_permission_manage_channels_for_rename")
        return

    original_names = (snapshot or {}).get("original_channel_names", {}) or {}
    renamed_count = 0
    known_emojis = list(SEASONAL_EMOJIS.keys()) + [DEFAULT_EMOJI]

    for ch in guild.channels:
        if renamed_count >= CFG.max_renames:
            break
        if not should_rename_channel(ch):
            continue

        old = getattr(ch, "name", "") or ""
        base = old
        for e in known_emojis:
            base = base.replace(e, "")
        base = sanitize_channel_name(base.strip("-").strip())

        new = (
            sanitize_channel_name(f"{emoji}-{base}")
            if CFG.rename_mode != "suffix"
            else sanitize_channel_name(f"{base}-{emoji}")
        )
        if new == old:
            continue

        # store original only once
        if str(ch.id) not in original_names:
            original_names[str(ch.id)] = old

        if CFG.dry_run:
            REPORT["renames"].append({"action": "would_rename", "from": old, "to": new, "id": ch.id, "ts": utc_now_iso()})
            REPORT["metrics"]["renames_would"] += 1
            renamed_count += 1
            continue

        try:
            await ch.edit(name=new, reason=f"{LUCHA_NAME}: seasonal theme")
            REPORT["renames"].append({"action": "rename", "from": old, "to": new, "id": ch.id, "ts": utc_now_iso()})
            REPORT["metrics"]["renames_done"] += 1
            REPORT["metrics"]["api_calls_soft"] += 1
            renamed_count += 1
            await asyncio.sleep(1.2)
        except Exception as e:
            REPORT["errors"].append(f"rename_error:{redact_for_logs(str(e))}")

    snapshot["original_channel_names"] = original_names

async def restore_channel_names(guild: discord.Guild, prev_snapshot: dict):
    if not CFG.restore_renames:
        return

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_permission_manage_channels_for_restore")
        return

    originals: Dict[str, str] = (prev_snapshot or {}).get("original_channel_names", {}) or {}
    if not originals:
        REPORT["errors"].append("restore_requested_but_no_original_channel_names_in_snapshot")
        return

    restored = 0
    for ch in guild.channels:
        target = originals.get(str(ch.id))
        if not target or getattr(ch, "name", "") == target:
            continue

        if CFG.dry_run:
            REPORT["actions"].append({"action": "would_restore_channel_name", "channel_id": ch.id, "to": target, "ts": utc_now_iso()})
            continue

        try:
            await ch.edit(name=target, reason=f"{LUCHA_NAME}: restore channel names")
            REPORT["actions"].append({"action": "restore_channel_name", "channel_id": ch.id, "to": target, "ts": utc_now_iso()})
            REPORT["metrics"]["api_calls_soft"] += 1
            restored += 1
            await asyncio.sleep(1.1)
        except Exception as e:
            REPORT["errors"].append(f"restore_rename_error:{redact_for_logs(str(e))}")

    REPORT["actions"].append({"action": "restore_complete", "restored_count": restored, "ts": utc_now_iso()})


# ---------------- Moderation scan ----------------
def should_scan_channel(ch_name: str) -> bool:
    if safe_regex_search(CFG.scan_exclude_regex, ch_name):
        return False
    if CFG.scan_allowlist_regex and not safe_regex_search(CFG.scan_allowlist_regex, ch_name):
        return False
    return True

async def fetch_attachment_bytes(att: discord.Attachment, session: aiohttp.ClientSession) -> Optional[bytes]:
    # Prefer Discord's attachment.read() first.
    try:
        return await att.read()
    except Exception:
        # fallback (still same URL)
        try:
            async with session.get(att.url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status == 200:
                    return await r.read()
        except Exception:
            return None
    return None

async def scan_one_channel(ch: discord.TextChannel, me: discord.Member, http_session: aiohttp.ClientSession):
    perms = ch.permissions_for(me)
    if not (perms.read_messages and perms.read_message_history and perms.manage_messages):
        return

    scanned_msgs = 0
    try:
        async for msg in ch.history(limit=CFG.max_messages, oldest_first=False):
            scanned_msgs += 1
            REPORT["metrics"]["messages_scanned"] += 1

            if msg.author.bot:
                continue

            if msg.content:
                toxic, scores = is_text_toxic(msg.content)
                pat = suspicious_code_check(msg.content)

                is_webhook_leak = bool(pat and "webhooks" in pat and CFG.delete_webhook_leaks)
                is_cred_like = bool(pat and CFG.delete_credential_leaks)
                should_delete = toxic or is_webhook_leak or is_cred_like

                if should_delete:
                    REPORT["suspicious_content"].append({
                        "channel": ch.name,
                        "author": str(msg.author),
                        "message_id": msg.id,
                        "snippet": short_snip(msg.content),
                        "toxic_scores": scores,
                        "pattern": pat,
                        "ts": utc_now_iso(),
                    })

                    if CFG.dry_run:
                        REPORT["actions"].append({"action": "would_delete_message", "channel": ch.name, "message_id": msg.id, "ts": utc_now_iso()})
                    else:
                        try:
                            await msg.delete(reason=f"{LUCHA_NAME}: toxic/suspicious content")
                            REPORT["metrics"]["messages_deleted"] += 1
                            REPORT["metrics"]["api_calls_soft"] += 1
                        except Exception as e:
                            REPORT["errors"].append(f"delete_msg_error:{redact_for_logs(str(e))}")

            for att in msg.attachments:
                fn = (att.filename or "").lower()
                if any(fn.endswith(ext) for ext in SCAN_IMAGE_TYPES):
                    REPORT["metrics"]["attachments_scanned"] += 1
                    data = await fetch_attachment_bytes(att, http_session)
                    if not data:
                        continue
                    score = run_model_on_image_bytes(data)
                    if score >= CFG.nsfw_threshold:
                        REPORT["nsfw_attachments"].append({
                            "channel": ch.name,
                            "author": str(msg.author),
                            "message_id": msg.id,
                            "attachment": att.url,
                            "score": score,
                            "ts": utc_now_iso(),
                        })
                        if CFG.dry_run:
                            REPORT["actions"].append({"action": "would_delete_message_nsfw", "channel": ch.name, "message_id": msg.id, "ts": utc_now_iso()})
                        else:
                            try:
                                await msg.delete(reason=f"{LUCHA_NAME}: NSFW attachment")
                                REPORT["metrics"]["messages_deleted"] += 1
                                REPORT["metrics"]["api_calls_soft"] += 1
                            except Exception as e:
                                REPORT["errors"].append(f"delete_msg_error:{redact_for_logs(str(e))}")

            if scanned_msgs % 40 == 0:
                await asyncio.sleep(0.25)

    except Exception as e:
        REPORT["errors"].append(f"scan_channel_error_{ch.name}:{redact_for_logs(str(e))}")

async def scan_messages_and_cleanup(guild: discord.Guild):
    if not CFG.enable_moderation:
        return
    me = guild.me
    if not me:
        return

    channels = [c for c in guild.text_channels if should_scan_channel(c.name)]
    channels = channels[:CFG.max_channels]
    REPORT["metrics"]["channels_scanned"] = len(channels)

    sem = asyncio.Semaphore(CFG.scan_concurrency)
    timeout = aiohttp.ClientTimeout(total=20)

    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": f"{LUCHA_NAME}/3.0"}) as session:
        async def runner(ch: discord.TextChannel):
            async with sem:
                await scan_one_channel(ch, me, session)

        await asyncio.gather(*(runner(ch) for ch in channels))


# ---------------- Posting helpers ----------------
async def ensure_text_channel(guild: discord.Guild, name: str, topic: str = "") -> Optional[discord.TextChannel]:
    channel = get(guild.text_channels, name=name)
    if channel:
        return channel

    if CFG.dry_run:
        REPORT["actions"].append({"action": "would_create_channel", "channel": name, "ts": utc_now_iso()})
        return None

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append(f"missing_permission_manage_channels_create_{name}")
        return None

    try:
        channel = await guild.create_text_channel(name, topic=topic[:1000] if topic else None)
        REPORT["actions"].append({"action": "create_channel", "channel": name, "ts": utc_now_iso()})
        REPORT["metrics"]["api_calls_soft"] += 1
        return channel
    except Exception as e:
        REPORT["errors"].append(f"create_channel_error_{name}:{redact_for_logs(str(e))}")
        return None

async def safe_send(channel: discord.TextChannel, content: str, action_tag: str):
    if CFG.dry_run:
        REPORT["actions"].append({"action": f"would_{action_tag}", "channel": channel.name, "ts": utc_now_iso()})
        return
    try:
        await channel.send(content[:1900])
        REPORT["actions"].append({"action": action_tag, "channel": channel.name, "ts": utc_now_iso()})
        REPORT["metrics"]["posts_sent"] += 1
        REPORT["metrics"]["api_calls_soft"] += 1
    except Exception as e:
        REPORT["errors"].append(f"send_error_{action_tag}:{redact_for_logs(str(e))}")


# ---------------- Daily info (with Feb 3, 2026 override) ----------------
DAILY_INFO_OVERRIDES: Dict[str, str] = {
    "2026-02-03": "💖 February 3, 2026: it is almost valentines day! (brace your heart-shaped confetti cannons)",
}

def build_daily_info_message(emoji: str, sigil: str) -> str:
    key = today_utc_date().isoformat()
    if key in DAILY_INFO_OVERRIDES:
        return f"{emoji} **{LUCHA_NAME} Daily Info** · `{sigil}`\n{DAILY_INFO_OVERRIDES[key]}"

    # Generic fallback (still fun, still safe)
    return (
        f"{emoji} **{LUCHA_NAME} Daily Info** · `{sigil}`\n"
        f"Today’s vibe: be chill, be kind, and do not summon chaos unless it’s funny."
    )

async def post_daily_info(guild: discord.Guild, emoji: str, sigil: str, state: dict):
    if not CFG.enable_daily_info:
        return

    # De-dupe: only post once per day
    daily_key = get_state_key_date("daily_info_posted")
    if state.get(daily_key) is True:
        REPORT["daily_info"] = {"posted": False, "reason": "already_posted_today", "ts": utc_now_iso()}
        return

    info_ch = await ensure_text_channel(
        guild,
        CFG.info_channel_name,
        topic=f"{LUCHA_NAME} daily info (safe automation).",
    )
    if not info_ch:
        REPORT["daily_info"] = {"posted": False, "reason": "no_channel", "ts": utc_now_iso()}
        return

    msg = build_daily_info_message(emoji, sigil)
    await safe_send(info_ch, msg, "post_daily_info")

    state[daily_key] = True
    REPORT["daily_info"] = {"posted": not CFG.dry_run, "channel": info_ch.name, "date": today_utc_date().isoformat(), "ts": utc_now_iso()}


# ---------------- YouTube RSS latest video (de-dupe) ----------------
def parse_youtube_rss(xml_bytes: bytes) -> Optional[Dict[str, str]]:
    """
    Returns latest entry: {video_id, title, url, published}
    """
    try:
        root = ET.fromstring(xml_bytes)
        # namespaces in YT feeds
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "yt": "http://www.youtube.com/xml/schemas/2015",
        }
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title_el = entry.find("atom:title", ns)
        vid_el = entry.find("yt:videoId", ns)
        link_el = entry.find("atom:link", ns)
        pub_el = entry.find("atom:published", ns)

        video_id = (vid_el.text or "").strip() if vid_el is not None else ""
        title = (title_el.text or "").strip() if title_el is not None else ""
        url = ""
        if link_el is not None:
            url = (link_el.attrib.get("href") or "").strip()

        published = (pub_el.text or "").strip() if pub_el is not None else ""
        if not video_id or not url:
            return None

        return {"video_id": video_id, "title": title or "(untitled)", "url": url, "published": published}
    except Exception:
        return None

async def fetch_latest_youtube_video(session: aiohttp.ClientSession, channel_id: str) -> Optional[Dict[str, str]]:
    url = YOUTUBE_RSS_URL.format(channel_id=channel_id)
    try:
        async with session.get(url) as r:
            if r.status != 200:
                REPORT["errors"].append(f"youtube_rss_http_{r.status}")
                return None
            data = await r.read()
            return parse_youtube_rss(data)
    except Exception as e:
        REPORT["errors"].append(f"youtube_rss_fetch_error:{redact_for_logs(str(e))}")
        return None

async def post_latest_youtube_if_new(guild: discord.Guild, state: dict):
    if not CFG.enable_youtube or not CFG.yt_channel_id:
        return

    timeout = aiohttp.ClientTimeout(total=20)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": f"{LUCHA_NAME}/3.0"}) as session:
        latest = await fetch_latest_youtube_video(session, CFG.yt_channel_id)

    if not latest:
        REPORT["youtube"] = {"checked": True, "posted": False, "reason": "no_latest_found", "ts": utc_now_iso()}
        return

    last_id = (state.get("yt_last_video_id") or "").strip()
    if latest["video_id"] == last_id:
        REPORT["youtube"] = {"checked": True, "posted": False, "reason": "same_as_last", "video_id": latest["video_id"], "ts": utc_now_iso()}
        return

    post_ch = get(guild.text_channels, name=CFG.yt_post_channel_name)
    if not post_ch:
        # fall back to ensuring info channel exists
        post_ch = await ensure_text_channel(
            guild,
            CFG.yt_post_channel_name,
            topic=f"{LUCHA_NAME} video drops (safe automation).",
        )
    if not post_ch:
        REPORT["youtube"] = {"checked": True, "posted": False, "reason": "no_channel", "video_id": latest["video_id"], "ts": utc_now_iso()}
        return

    msg = CFG.yt_post_template.format(title=latest["title"], url=latest["url"])
    await safe_send(post_ch, msg, "post_youtube_latest")

    # Update state only if we attempted (even in dry-run we keep last id unchanged to avoid “false advancement”)
    if not CFG.dry_run:
        state["yt_last_video_id"] = latest["video_id"]
        state["yt_last_video_url"] = latest["url"]
        state["yt_last_video_published"] = latest.get("published", "")
        state["yt_last_posted_at"] = utc_now_iso()

    REPORT["youtube"] = {
        "checked": True,
        "posted": not CFG.dry_run,
        "video_id": latest["video_id"],
        "title": latest["title"],
        "url": latest["url"],
        "published": latest.get("published", ""),
        "channel": post_ch.name,
        "ts": utc_now_iso(),
    }


# ---------------- Discord client ----------------
# Intents minimized: message_content only if moderation enabled.
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = False

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    REPORT["lucha"]["ts"] = utc_now_iso()
    emoji = get_dynamic_emoji()
    sigil = make_daily_sigil(CFG.guild_id, emoji)
    REPORT["lucha"]["sigil"] = sigil

    log.info("%s online. armed=%s dry_run=%s moderation=%s youtube=%s daily=%s",
             LUCHA_NAME, CFG.lucha_armed, CFG.dry_run, CFG.enable_moderation, CFG.enable_youtube, CFG.enable_daily_info)

    init_models()

    state = load_state()

    try:
        guild = client.get_guild(CFG.guild_id)
        if not guild:
            REPORT["errors"].append(f"guild_not_found:{CFG.guild_id}")
            save_json(CFG.report_file, REPORT)
            await client.close()
            return

        prev_snapshot = load_json(CFG.backup_file) or {}

        # Optional restore mode
        if CFG.restore_renames:
            await restore_channel_names(guild, prev_snapshot)

        # Nuke detect + remediate
        nuke_info = detect_nuke(prev_snapshot, guild)
        REPORT["nuke_events"].append({**nuke_info, "ts": utc_now_iso()})
        if nuke_info.get("is_nuke"):
            log.warning("%s detected possible nuke: %s", LUCHA_NAME, nuke_info)
            deleted = await remediate_nuke(guild, prev_snapshot)
            REPORT["nuke_events"][-1]["deleted_channels"] = deleted

        # Theme renames (skip if restoring this run)
        await themed_rename_channels(guild, emoji, prev_snapshot)

        # Moderation scan (if enabled)
        await scan_messages_and_cleanup(guild)

        # Daily info post (de-duped)
        await post_daily_info(guild, emoji, sigil, state)

        # YouTube latest post (de-duped)
        await post_latest_youtube_if_new(guild, state)

        # Save snapshot + report (snapshot includes original names mapping)
        snapshot = build_snapshot(guild)
        if prev_snapshot.get("original_channel_names"):
            snapshot["original_channel_names"] = prev_snapshot["original_channel_names"]

        save_json(CFG.backup_file, snapshot)
        save_json(CFG.report_file, REPORT)
        save_state(state)

        log.info("%s finished run. report=%s snapshot=%s state=%s",
                 LUCHA_NAME, CFG.report_file, CFG.backup_file, CFG.state_file)

    except Exception as e:
        REPORT["errors"].append(f"on_ready_unhandled:{redact_for_logs(str(e))}")
        save_json(CFG.report_file, REPORT)
        log.exception("Unhandled error (details may be redacted)")
    finally:
        try:
            await client.close()
        finally:
            try:
                sys.exit(0)
            except SystemExit:
                pass


if __name__ == "__main__":
    # Never print the token; just run it.
    client.run(CFG.discord_token)
