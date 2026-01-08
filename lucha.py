#!/usr/bin/env python3
"""
LuCha — Discord Scanner + Daily Word Ritual

What LuCha does:
- Conservative moderation (toxicity heuristic/model, NSFW image model if available, credential/webhook leaks, suspicious code)
- Nuke detection (snapshot compare) + cautious remediation of obvious foreign channels after a raid
- Daily "Word of the Day" pulled from Wikipedia (random page title + short summary, filtered to be more "word-like")
- Seasonal holiday theming: renames channels with an emoji prefix/suffix (safe + rate-limited + reversible)
- Writes JSON report + snapshot for auditing

Safety & operational controls (ENV):
Required:
  - DISCORD_TOKEN, GUILD_ID

Core files:
  - BACKUP_FILE (default: guild_backup.json)
  - REPORT_FILE (default: mod_report.json)

Safety switches:
  - LUCHA_ARMED=0/1              (default 0) if 0 -> forces DRY_RUN=True
  - DRY_RUN=0/1                  (default 0) if 1 -> never edit/delete/post, only report actions
  - ENABLE_RENAME=0/1            (default 1)
  - ENABLE_MODERATION=0/1        (default 1)
  - ENABLE_NUKE_REMEDIATION=0/1  (default 1)
  - ENABLE_WOTD=0/1              (default 1)

Renaming controls:
  - RENAME_MODE=prefix|suffix (default prefix)
  - MAX_RENAMES (default 40)
  - RENAME_EXCLUDE_REGEX (default empty)
  - RENAME_ALLOWLIST_REGEX (default empty)   # if set, ONLY rename channels matching this regex
  - RENAME_INCLUDE_CATEGORIES=0/1 (default 0)

Scanning controls:
  - MAX_MESSAGES (default 250)
  - TOXIC_THRESHOLD (default 0.6)
  - NSFW_THRESHOLD (default 0.8)
  - SCAN_ALLOWLIST_REGEX (default empty)     # if set, ONLY scan channels matching this regex
  - SCAN_EXCLUDE_REGEX (default empty)       # skip scanning channels matching this regex
  - DELETE_WEBHOOK_LEAKS=0/1 (default 1)
  - DELETE_CREDENTIAL_LEAKS=0/1 (default 1)

Nuke detection controls:
  - NUKE_DELETION_THRESHOLD (default 0.15)  # fraction of trusted channels missing
  - NUKE_NEW_CHANNEL_THRESHOLD (default 6)  # new channels created to consider attack

Posting:
  - INFO_CHANNEL_NAME (default "daily-info")
"""

import os
import sys
import json
import io
import re
import asyncio
import random
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional, List

import aiohttp
import discord
from discord.utils import get

# Optional: load env from .env (handy locally)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional ML deps (graceful)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForImageClassification
    import torch
    TF_AVAILABLE = True
except Exception:
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoFeatureExtractor = None
    AutoModelForImageClassification = None
    torch = None
    TF_AVAILABLE = False

# ---------------- CONFIG ----------------
LUCHA_NAME = "LuCha"

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None

BACKUP_FILE = os.getenv("BACKUP_FILE", "guild_backup.json")
REPORT_FILE = os.getenv("REPORT_FILE", "mod_report.json")
INFO_CHANNEL_NAME = os.getenv("INFO_CHANNEL_NAME", "daily-info")

MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "250"))
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.6"))
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.8"))

LUCHA_ARMED = os.getenv("LUCHA_ARMED", "0").strip() == "1"
DRY_RUN = os.getenv("DRY_RUN", "0").strip() == "1"
if not LUCHA_ARMED:
    DRY_RUN = True  # hard safety: must explicitly arm for destructive edits

ENABLE_RENAME = os.getenv("ENABLE_RENAME", "1").strip() == "1"
ENABLE_MODERATION = os.getenv("ENABLE_MODERATION", "1").strip() == "1"
ENABLE_NUKE_REMEDIATION = os.getenv("ENABLE_NUKE_REMEDIATION", "1").strip() == "1"
ENABLE_WOTD = os.getenv("ENABLE_WOTD", "1").strip() == "1"

RENAME_MODE = os.getenv("RENAME_MODE", "prefix").strip().lower()  # prefix|suffix
MAX_RENAMES = int(os.getenv("MAX_RENAMES", "40"))
RENAME_EXCLUDE_REGEX = os.getenv("RENAME_EXCLUDE_REGEX", "").strip()
RENAME_ALLOWLIST_REGEX = os.getenv("RENAME_ALLOWLIST_REGEX", "").strip()
RENAME_INCLUDE_CATEGORIES = os.getenv("RENAME_INCLUDE_CATEGORIES", "0").strip() == "1"

SCAN_ALLOWLIST_REGEX = os.getenv("SCAN_ALLOWLIST_REGEX", "").strip()
SCAN_EXCLUDE_REGEX = os.getenv("SCAN_EXCLUDE_REGEX", "").strip()

DELETE_WEBHOOK_LEAKS = os.getenv("DELETE_WEBHOOK_LEAKS", "1").strip() == "1"
DELETE_CREDENTIAL_LEAKS = os.getenv("DELETE_CREDENTIAL_LEAKS", "1").strip() == "1"

NUKE_DELETION_THRESHOLD = float(os.getenv("NUKE_DELETION_THRESHOLD", "0.15"))
NUKE_NEW_CHANNEL_THRESHOLD = int(os.getenv("NUKE_NEW_CHANNEL_THRESHOLD", "6"))

SCAN_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

# Seasonal emoji theme
SEASONAL_EMOJIS = {"🎃": [10, 11], "🎄": [12], "💖": [2], "🌸": [3, 4]}
DEFAULT_EMOJI = "🔹"

SUSPICIOUS_PATTERNS = [
    # credential-ish
    r"(api_key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----",

    # code exec
    r"\bexec\(",
    r"\beval\(",
    r"base64\.b64decode",
    r"powershell\s+.*-EncodedCommand",
    r"\bcurl\s+.*--output\b",
    r"\bwget\s+.*\s-O\b",

    # discord webhooks
    r"(discord(app)?\.com\/api\/webhooks\/\d+\/[A-Za-z0-9_\-]+)",
]

# ---------------- REPORT ----------------
REPORT: Dict[str, object] = {
    "lucha": {
        "name": LUCHA_NAME,
        "armed": LUCHA_ARMED,
        "dry_run": DRY_RUN,
        "ts": None,
        "sigil": None,
    },
    "nuke_events": [],
    "restoration_actions": [],
    "renames": [],
    "toxic_messages": [],
    "nsfw_attachments": [],
    "suspicious_content": [],
    "word_of_the_day": {},
    "errors": [],
}

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("LuCha")

# ---------------- ML models (optional) ----------------
txt_model = None
txt_tokenizer = None
img_extractor = None
img_model = None

def init_models():
    global txt_model, txt_tokenizer, img_extractor, img_model

    # Toxicity
    try:
        if AutoTokenizer and AutoModelForSequenceClassification:
            txt_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
            txt_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
            log.info("LuCha loaded toxic text model.")
    except Exception as e:
        REPORT["errors"].append(f"toxic_model_load_error:{e}")

    # NSFW image
    try:
        if AutoFeatureExtractor and AutoModelForImageClassification:
            img_extractor = AutoFeatureExtractor.from_pretrained("Falconsai/nsfw_image_detection")
            img_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
            log.info("LuCha loaded NSFW image model.")
    except Exception as e:
        REPORT["errors"].append(f"nsfw_model_load_error:{e}")

# ---------------- Utilities ----------------
def utc_now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def load_json(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        REPORT["errors"].append(f"load_json_error_{path}:{e}")
    return {}

def save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        REPORT["errors"].append(f"save_json_error_{path}:{e}")

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

def short_snip(s: str, n: int = 260) -> str:
    s = s or ""
    return (s[:n] + "...") if len(s) > n else s

def safe_regex_search(pattern: str, text: str) -> bool:
    if not pattern:
        return False
    try:
        return bool(re.search(pattern, text, re.IGNORECASE))
    except re.error:
        # bad user regex: ignore
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

# ---------------- Wikipedia Word of the Day ----------------
WIKI_RANDOM_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/random/summary"

def is_good_wiki_title(title: str) -> bool:
    """Filter out super-janky titles so WotD feels more like a 'word'."""
    if not title:
        return False
    if len(title) > 42:
        return False
    if ":" in title:  # filter Category:, Portal:, etc.
        return False
    if title.startswith("List of "):
        return False
    # avoid pure years/numbers
    if re.fullmatch(r"\d{3,4}", title.strip()):
        return False
    return True

async def fetch_json_with_retries(url: str, session: aiohttp.ClientSession, retries: int = 3, timeout_s: int = 15) -> Optional[dict]:
    for attempt in range(retries):
        try:
            async with session.get(url, timeout=timeout_s, headers={"User-Agent": f"{LUCHA_NAME}/1.0"}) as r:
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}")
                return await r.json()
        except Exception as e:
            if attempt == retries - 1:
                REPORT["errors"].append(f"fetch_json_error:{url}:{e}")
                return None
            await asyncio.sleep(0.7 + attempt * 0.9)

async def get_wikipedia_word_of_the_day() -> Dict[str, str]:
    async with aiohttp.ClientSession() as session:
        # Try multiple random pages until we get a decent title
        for _ in range(6):
            data = await fetch_json_with_retries(WIKI_RANDOM_SUMMARY, session)
            if not data:
                break
            title = (data.get("title") or "").strip()
            extract = (data.get("extract") or "").strip()
            url = ""
            try:
                url = data.get("content_urls", {}).get("desktop", {}).get("page", "") or ""
            except Exception:
                url = ""
            if is_good_wiki_title(title) and extract:
                return {"word": title, "definition": extract[:650], "url": url}

    # fallback
    return {"word": "Serendipity", "definition": "A pleasant surprise… (Wikipedia fetch failed today.)", "url": ""}

def build_wotd_message(emoji: str, sigil: str, w: Dict[str, str]) -> str:
    mood_lines = {
        "🎃": "Server mood: spooky-cozy. No jump-scares unless they’re funny.",
        "🎄": "Server mood: jingle-powered. Be nice or Santa’s logging it.",
        "💖": "Server mood: wholesome chaos. Compliments are buff spells.",
        "🌸": "Server mood: springy and bright. Touch grass (in-game only).",
        "🔹": "Server mood: classic chill. Hydrate and don’t feed the trolls.",
    }
    mood = mood_lines.get(emoji, mood_lines["🔹"])
    url_part = f"\nMore: {w['url']}" if w.get("url") else ""
    return (
        f"{emoji} **{LUCHA_NAME} Daily Rune**  ·  `{sigil}`\n"
        f"**Wikipedia Word of the Day:** **{w.get('word','(unknown)')}**\n"
        f"{w.get('definition','(No summary available.)')}\n\n"
        f"_{mood}_{url_part}"
    )

# ---------------- Moderation models ----------------
def run_model_on_text(text: str) -> Dict[str, float]:
    if not txt_model or not txt_tokenizer or not torch:
        # heuristic fallback
        lower = (text or "").lower()
        bad = ["idiot", "kill", "stfu", "die", "racist", "nazi", "faggot", "retard"]
        score = 0.0
        for b in bad:
            if b in lower:
                score += 0.25
        return {"toxic": min(score, 1.0)}
    try:
        inputs = txt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = txt_model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy()
        return {"toxic": float(probs[0])}
    except Exception as e:
        REPORT["errors"].append(f"text_scan_error:{e}")
        return {"toxic": 0.0}

def is_text_toxic(text: str) -> Tuple[bool, Dict[str, float]]:
    scores = run_model_on_text(text)
    return (float(scores.get("toxic", 0.0)) >= TOXIC_THRESHOLD, scores)

def run_model_on_image_bytes(data: bytes) -> float:
    if not img_extractor or not img_model or not torch:
        return 0.0
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = img_extractor(images=img, return_tensors="pt")
        outputs = img_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
        # many models: [safe, nsfw]
        if len(probs) > 1:
            return float(probs[1])
        return float(max(probs))
    except Exception as e:
        REPORT["errors"].append(f"image_scan_error:{e}")
        return 0.0

# ---------------- Snapshot & nuke detection ----------------
def build_snapshot(guild: discord.Guild) -> dict:
    snapshot = {
        "ts": utc_now_iso(),
        "channels": [],
        "original_channel_names": {},  # id -> original name (for reversible renames)
    }
    for ch in guild.channels:
        snapshot["channels"].append({
            "id": ch.id,
            "name": ch.name,
            "type": str(ch.type),
            "position": getattr(ch, "position", None),
        })
        snapshot["original_channel_names"][str(ch.id)] = ch.name
    return snapshot

def detect_nuke(prev_snapshot: dict, guild: discord.Guild) -> dict:
    prev_channels = prev_snapshot.get("channels", []) if prev_snapshot else []
    prev_names = [c.get("name") for c in prev_channels if c.get("name")]
    curr_names = [c.name for c in guild.channels]

    missing = [n for n in prev_names if n not in curr_names]
    new = [c.name for c in guild.channels if c.name not in prev_names]

    deleted_frac = (len(missing) / max(1, len(prev_names))) if prev_names else 0.0
    is_nuke = bool(prev_names) and (deleted_frac >= NUKE_DELETION_THRESHOLD) and (len(new) >= NUKE_NEW_CHANNEL_THRESHOLD)

    return {
        "is_nuke": is_nuke,
        "deleted_count": len(missing),
        "created_count": len(new),
        "deleted_frac": round(deleted_frac, 4),
        "missing_sample": missing[:10],
        "new_sample": new[:10],
    }

async def remediate_nuke(guild: discord.Guild, prev_snapshot: dict):
    if not ENABLE_NUKE_REMEDIATION:
        return []

    prev_names = {c.get("name") for c in (prev_snapshot.get("channels", []) if prev_snapshot else [])}
    deleted = []

    # conservative: only delete channels with very suspicious names OR if channel count exploded
    exploded = len(guild.channels) > (len(prev_names) + 8)

    for ch in guild.channels:
        if ch.name in prev_names:
            continue

        suspect_name = bool(re.search(r"(spam|raid|hacked|nuke|giveaway|free-.*|bot-)", ch.name, re.IGNORECASE))
        if not (suspect_name or exploded):
            continue

        if DRY_RUN:
            REPORT["restoration_actions"].append({"action": "would_delete_channel", "channel": ch.name, "ts": utc_now_iso()})
            deleted.append(ch.name)
            continue

        try:
            await ch.delete(reason=f"{LUCHA_NAME}: foreign channel after nuke (conservative)")
            REPORT["restoration_actions"].append({"action": "delete_channel", "channel": ch.name, "ts": utc_now_iso()})
            deleted.append(ch.name)
            await asyncio.sleep(1.0)
        except Exception as e:
            REPORT["errors"].append(f"delete_channel_error_{ch.name}:{e}")

    return deleted

# ---------------- Channel rename theming ----------------
def should_rename_channel(ch: discord.abc.GuildChannel) -> bool:
    name = getattr(ch, "name", "") or ""
    if safe_regex_search(RENAME_EXCLUDE_REGEX, name):
        return False
    if RENAME_ALLOWLIST_REGEX and not safe_regex_search(RENAME_ALLOWLIST_REGEX, name):
        return False
    # skip categories unless enabled
    if isinstance(ch, discord.CategoryChannel) and not RENAME_INCLUDE_CATEGORIES:
        return False
    return True

async def themed_rename_channels(guild: discord.Guild, emoji: str, prev_snapshot: dict):
    if not ENABLE_RENAME:
        return

    me = guild.me
    if not me or not guild.me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_permission_manage_channels")
        return

    original_names = (prev_snapshot or {}).get("original_channel_names", {}) or {}
    renamed_count = 0

    # strip any seasonal emoji from channel name to prevent stacking
    known_emojis = list(SEASONAL_EMOJIS.keys()) + [DEFAULT_EMOJI]

    for ch in guild.channels:
        if renamed_count >= MAX_RENAMES:
            break
        if not should_rename_channel(ch):
            continue

        old = ch.name

        base = old
        for e in known_emojis:
            base = base.replace(e, "")
        base = sanitize_channel_name(base.strip("-").strip())

        new = sanitize_channel_name(f"{emoji}-{base}") if RENAME_MODE != "suffix" else sanitize_channel_name(f"{base}-{emoji}")
        if new == old:
            continue

        if str(ch.id) not in original_names:
            original_names[str(ch.id)] = old

        if DRY_RUN:
            REPORT["renames"].append({"action": "would_rename", "from": old, "to": new, "id": ch.id, "ts": utc_now_iso()})
            renamed_count += 1
            continue

        try:
            await ch.edit(name=new, reason=f"{LUCHA_NAME}: seasonal theme")
            REPORT["renames"].append({"action": "rename", "from": old, "to": new, "id": ch.id, "ts": utc_now_iso()})
            renamed_count += 1
            await asyncio.sleep(1.25)  # gentle rate limit
        except Exception as e:
            REPORT["errors"].append(f"rename_error_{old}:{e}")

    if prev_snapshot is not None:
        prev_snapshot["original_channel_names"] = original_names

# ---------------- Moderation scan ----------------
def should_scan_channel(ch_name: str) -> bool:
    if safe_regex_search(SCAN_EXCLUDE_REGEX, ch_name):
        return False
    if SCAN_ALLOWLIST_REGEX and not safe_regex_search(SCAN_ALLOWLIST_REGEX, ch_name):
        return False
    return True

async def scan_messages_and_cleanup(guild: discord.Guild):
    if not ENABLE_MODERATION:
        return

    me = guild.me
    if not me:
        return

    for ch in guild.text_channels:
        if not should_scan_channel(ch.name):
            continue

        perms = ch.permissions_for(me)
        if not (perms.read_messages and perms.read_message_history and perms.manage_messages):
            continue

        try:
            async for msg in ch.history(limit=MAX_MESSAGES, oldest_first=False):
                if msg.author.bot:
                    continue

                # Text checks
                if msg.content:
                    toxic, scores = is_text_toxic(msg.content)

                    pat = suspicious_code_check(msg.content)
                    is_webhook_leak = bool(pat and "webhooks" in pat and DELETE_WEBHOOK_LEAKS)
                    is_cred_leak = bool(pat and ("api_key" in pat.lower() or "private key" in pat.lower() or "token" in pat.lower()) and DELETE_CREDENTIAL_LEAKS)

                    should_delete = toxic or is_webhook_leak or is_cred_leak or (pat is not None and ("exec" in pat or "eval" in pat))

                    if should_delete:
                        REPORT["suspicious_content"].append({
                            "channel": ch.name,
                            "author": str(msg.author),
                            "snippet": short_snip(msg.content),
                            "toxic_scores": scores,
                            "pattern": pat,
                            "ts": utc_now_iso(),
                        })
                        if DRY_RUN:
                            REPORT["restoration_actions"].append({"action": "would_delete_message", "channel": ch.name, "ts": utc_now_iso()})
                        else:
                            try:
                                await msg.delete(reason=f"{LUCHA_NAME}: toxic/suspicious content")
                            except Exception as e:
                                REPORT["errors"].append(f"delete_msg_error:{e}")

                # Attachment checks (images)
                for att in msg.attachments:
                    if any(att.filename.lower().endswith(ext) for ext in SCAN_IMAGE_TYPES):
                        data = None
                        try:
                            data = await att.read()
                        except Exception:
                            # fallback fetch
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(att.url, timeout=12) as r:
                                        if r.status == 200:
                                            data = await r.read()
                            except Exception:
                                data = None

                        if data:
                            score = run_model_on_image_bytes(data)
                            if score >= NSFW_THRESHOLD:
                                REPORT["nsfw_attachments"].append({
                                    "channel": ch.name,
                                    "author": str(msg.author),
                                    "attachment": att.url,
                                    "score": score,
                                    "ts": utc_now_iso(),
                                })
                                if DRY_RUN:
                                    REPORT["restoration_actions"].append({"action": "would_delete_message_nsfw", "channel": ch.name, "ts": utc_now_iso()})
                                else:
                                    try:
                                        await msg.delete(reason=f"{LUCHA_NAME}: NSFW attachment")
                                    except Exception as e:
                                        REPORT["errors"].append(f"delete_msg_error:{e}")

        except Exception as e:
            REPORT["errors"].append(f"scan_channel_error_{ch.name}:{e}")

# ---------------- Daily info channel ----------------
async def ensure_info_channel(guild: discord.Guild) -> Optional[discord.TextChannel]:
    channel = get(guild.text_channels, name=INFO_CHANNEL_NAME)
    if channel:
        return channel

    if DRY_RUN:
        REPORT["restoration_actions"].append({"action": "would_create_channel", "channel": INFO_CHANNEL_NAME, "ts": utc_now_iso()})
        return None

    try:
        channel = await guild.create_text_channel(
            INFO_CHANNEL_NAME,
            topic=f"{LUCHA_NAME} daily rune (Wikipedia Word of the Day + server mood).",
        )
        REPORT["restoration_actions"].append({"action": "create_channel", "channel": INFO_CHANNEL_NAME, "ts": utc_now_iso()})
        return channel
    except Exception as e:
        REPORT["errors"].append(f"create_channel_error:{e}")
        return None

# ---------------- Discord client ----------------
if not DISCORD_TOKEN or not GUILD_ID:
    print("DISCORD_TOKEN or GUILD_ID not set", file=sys.stderr)
    sys.exit(2)

intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    REPORT["lucha"]["ts"] = utc_now_iso()
    log.info("%s online. armed=%s dry_run=%s", LUCHA_NAME, LUCHA_ARMED, DRY_RUN)

    init_models()

    try:
        guild = client.get_guild(GUILD_ID)
        if not guild:
            REPORT["errors"].append(f"guild_not_found:{GUILD_ID}")
            await client.close()
            return

        emoji = get_dynamic_emoji()
        sigil = make_daily_sigil(GUILD_ID, emoji)
        REPORT["lucha"]["sigil"] = sigil

        # Load snapshot
        prev_snapshot = load_json(BACKUP_FILE) or {}

        # Detect nuke + remediate
        nuke_info = detect_nuke(prev_snapshot, guild)
        REPORT["nuke_events"].append({**nuke_info, "ts": utc_now_iso()})

        if nuke_info.get("is_nuke"):
            log.warning("%s detected possible nuke: %s", LUCHA_NAME, nuke_info)
            deleted = await remediate_nuke(guild, prev_snapshot)
            REPORT["nuke_events"][-1]["deleted_channels"] = deleted

        # Theme renames
        await themed_rename_channels(guild, emoji, prev_snapshot)

        # Moderation scan
        await scan_messages_and_cleanup(guild)

        # Word of the Day post
        if ENABLE_WOTD:
            info_ch = await ensure_info_channel(guild)
            w = await get_wikipedia_word_of_the_day()
            REPORT["word_of_the_day"] = {**w, "sigil": sigil, "emoji": emoji, "ts": utc_now_iso()}

            if info_ch:
                msg = build_wotd_message(emoji, sigil, w)
                if DRY_RUN:
                    REPORT["restoration_actions"].append({"action": "would_post_wotd", "channel": info_ch.name, "ts": utc_now_iso()})
                else:
                    try:
                        await info_ch.send(msg[:1900])
                        REPORT["restoration_actions"].append({"action": "post_wotd", "channel": info_ch.name, "ts": utc_now_iso()})
                    except Exception as e:
                        REPORT["errors"].append(f"post_wotd_error:{e}")

        # Save snapshot + report
        snapshot = build_snapshot(guild)
        # Preserve original names map (so we can reverse later if you add a restore mode)
        if prev_snapshot.get("original_channel_names"):
            snapshot["original_channel_names"] = prev_snapshot["original_channel_names"]

        save_json(BACKUP_FILE, snapshot)
        save_json(REPORT_FILE, REPORT)

        log.info("%s finished daily run. report=%s snapshot=%s", LUCHA_NAME, REPORT_FILE, BACKUP_FILE)

    except Exception as e:
        REPORT["errors"].append(f"on_ready_unhandled:{e}")
        save_json(REPORT_FILE, REPORT)
        log.exception("Unhandled error")

    finally:
        try:
            await client.close()
        finally:
            try:
                sys.exit(0)
            except SystemExit:
                pass

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
