#!/usr/bin/env python3
"""
daily_scan_rewrite.py (holiday-word edition)

Discord moderation + immersive daily flavor bot:
- Conservative mod actions (toxic/NSFW/credential leaks/malicious code)
- Detect nuke-like behavior by comparing snapshots
- Remediate foreign channels cautiously after nuke
- Post "Word of the Day" from Wikipedia (random page title + summary)
- Rename channels with seasonal emoji theme (safe, rate-limited, reversible)
- Snapshot/backup used as "trusted" state + to restore original names
- REPORT JSON output of actions taken

CONFIG via environment variables:
- DISCORD_TOKEN, GUILD_ID
- BACKUP_FILE (optional) default "guild_backup.json"
- REPORT_FILE (optional) default "mod_report.json"

Optional safety toggles:
- DRY_RUN=1              -> do not delete/rename/create, only report
- ENABLE_RENAME=1        -> allow renaming channels
- RENAME_MODE=prefix     -> prefix | suffix
- MAX_RENAMES=50         -> cap rename operations per run
- RENAME_EXCLUDE_REGEX   -> regex to skip channels (e.g. "^(rules|announcements)$")
"""

import os
import sys
import json
import io
import asyncio
import random
import re
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
import logging

import aiohttp
import discord
from discord.utils import get

# Optional ML dependencies (try to import; fallback gracefully)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForImageClassification
    import torch
    TF_AVAILABLE = True
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoFeatureExtractor = None
    AutoModelForImageClassification = None
    torch = None
    TF_AVAILABLE = False

# Embeddings (not used now, but kept if you want later)
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    np = None
    ST_AVAILABLE = False

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None

BACKUP_FILE = os.getenv("BACKUP_FILE", "guild_backup.json")
REPORT_FILE = os.getenv("REPORT_FILE", "mod_report.json")

MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "300"))
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.6"))
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.75"))

DRY_RUN = os.getenv("DRY_RUN", "0").strip() == "1"

ENABLE_RENAME = os.getenv("ENABLE_RENAME", "1").strip() == "1"
RENAME_MODE = os.getenv("RENAME_MODE", "prefix").strip().lower()  # prefix|suffix
MAX_RENAMES = int(os.getenv("MAX_RENAMES", "50"))
RENAME_EXCLUDE_REGEX = os.getenv("RENAME_EXCLUDE_REGEX", "").strip()

# Nuke detection tunables
NUKE_DELETION_THRESHOLD = float(os.getenv("NUKE_DELETION_THRESHOLD", "0.15"))
NUKE_NEW_CHANNEL_THRESHOLD = int(os.getenv("NUKE_NEW_CHANNEL_THRESHOLD", "6"))

SCAN_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

SUSPICIOUS_PATTERNS = [
    r"(api_key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----",
    r"exec\(",
    r"eval\(",
    r"base64\.b64decode",
    r"curl .* --output",
    r"powershell .* -EncodedCommand",
    r"wget .* -O",
    r"(discord(app)?\.com\/api\/webhooks\/\d+\/[A-Za-z0-9_\-]+)",
]

# Seasonal emoji theme
SEASONAL_EMOJIS = {"🎃": [10, 11], "🎄": [12], "💖": [2], "🌸": [3, 4]}
DEFAULT_EMOJI = "🔹"

REPORT: Dict[str, object] = {
    "toxic_messages": [],
    "nsfw_attachments": [],
    "suspicious_code": [],
    "nuke_events": [],
    "restoration_actions": [],
    "word_of_the_day": {},
    "renames": [],
    "decorations": [],
    "errors": [],
    "dry_run": DRY_RUN,
}

if not DISCORD_TOKEN or not GUILD_ID:
    print("DISCORD_TOKEN or GUILD_ID not set", file=sys.stderr)
    sys.exit(2)

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("daily_scan_rewrite")

# ---------------- ML model setup (graceful) ----------------
txt_model = None
txt_tokenizer = None
img_extractor = None
img_model = None

def init_models():
    global txt_model, txt_tokenizer, img_extractor, img_model

    # Text toxicity model
    try:
        if AutoTokenizer and AutoModelForSequenceClassification:
            txt_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
            txt_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
            log.info("Loaded toxic-bert model.")
    except Exception as e:
        log.warning("Could not load toxic text model: %s", e)
        REPORT["errors"].append(f"toxic_model_load_error: {str(e)}")

    # Image NSFW detection
    try:
        if AutoFeatureExtractor and AutoModelForImageClassification:
            img_extractor = AutoFeatureExtractor.from_pretrained("Falconsai/nsfw_image_detection")
            img_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
            log.info("Loaded nsfw image model.")
    except Exception as e:
        log.warning("Could not load nsfw image model: %s", e)
        REPORT["errors"].append(f"nsfw_model_load_error: {str(e)}")

# ---------------- UTILITIES ----------------
def load_json(path: str):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning("Failed loading json %s: %s", path, e)
    return {}

def save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Failed saving json %s: %s", path, e)

def utc_now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def get_dynamic_emoji():
    month = datetime.utcnow().month
    for e, months in SEASONAL_EMOJIS.items():
        if month in months:
            return e
    return DEFAULT_EMOJI

def suspicious_code_check(text: str):
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    return None

def short_snip(s: str, n=300):
    return (s[:n] + "...") if len(s) > n else s

def sanitize_channel_name(name: str) -> str:
    """
    Discord channel names must be <=100 chars and typically lowercase with hyphens.
    We'll keep the existing name style, but enforce length and strip weird whitespace.
    """
    name = re.sub(r"\s+", "-", name.strip())
    name = re.sub(r"-{2,}", "-", name)
    name = name.strip("-")
    if len(name) < 1:
        name = "channel"
    return name[:100]

def channel_should_skip_rename(ch_name: str) -> bool:
    if not RENAME_EXCLUDE_REGEX:
        return False
    try:
        return bool(re.search(RENAME_EXCLUDE_REGEX, ch_name, re.IGNORECASE))
    except re.error:
        # If user provided a bad regex, fail open (do not skip)
        return False

# ---------------- WIKIPEDIA WORD OF THE DAY ----------------
WIKI_RANDOM_URL = "https://en.wikipedia.org/api/rest_v1/page/random/summary"

async def fetch_wikipedia_word_of_the_day() -> Dict[str, str]:
    """
    Uses Wikipedia REST endpoint to get a random page summary.
    Returns dict with: title, extract, url.
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(WIKI_RANDOM_URL, timeout=15) as r:
                data = await r.json()
                title = (data.get("title") or "").strip()
                extract = (data.get("extract") or "").strip()
                url = ""
                content_urls = data.get("content_urls") or {}
                desktop = content_urls.get("desktop") or {}
                url = desktop.get("page") or ""
                # Treat title as the "word" (fun random topic)
                if not title:
                    raise ValueError("No title returned")
                return {"word": title, "definition": extract[:650], "url": url}
        except Exception as e:
            REPORT["errors"].append(f"wiki_fetch_error:{str(e)}")
            return {"word": "Serendipity", "definition": "A pleasant surprise… (Wikipedia fetch failed today.)", "url": ""}

def build_immersive_wotd_message(emoji: str, w: Dict[str, str]) -> str:
    mood_lines = {
        "🎃": "Server mood: spooky-but-cozy. No jump-scares unless they’re funny.",
        "🎄": "Server mood: jingle-powered. Be nice or Santa’s logging it.",
        "💖": "Server mood: wholesome chaos. Compliments are buff spells.",
        "🌸": "Server mood: springy and bright. Touch grass (in-game only).",
        "🔹": "Server mood: classic chill. Hydrate and don’t feed the trolls.",
    }
    mood = mood_lines.get(emoji, mood_lines["🔹"])
    url_part = f"\nMore: {w['url']}" if w.get("url") else ""
    definition = w.get("definition") or "(No summary available.)"
    return (
        f"{emoji} **Wikipedia Word of the Day**\n"
        f"**{w.get('word','(unknown)')}**\n"
        f"{definition}\n\n"
        f"_{mood}_{url_part}"
    )

# ---------------- NSFW IMAGE SCAN ----------------
def run_model_on_image_bytes(data: bytes) -> float:
    if not img_extractor or not img_model or not torch:
        return 0.0
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = img_extractor(images=img, return_tensors="pt")
        outputs = img_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
        # Many classifiers are [safe, nsfw]; if uncertain, be conservative
        if len(probs) > 1:
            return float(probs[1])
        return float(max(probs))
    except Exception as e:
        REPORT["errors"].append(f"image_scan_error:{str(e)}")
        return 0.0

# ---------------- TEXT TOXICITY SCAN ----------------
def run_model_on_text(text: str) -> Dict[str, float]:
    if not txt_tokenizer or not txt_model or not torch:
        # fallback heuristic: presence of harsh words
        lower = text.lower()
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
        REPORT["errors"].append(f"text_scan_error:{str(e)}")
        return {"toxic": 0.0}

def is_text_toxic(text: str) -> Tuple[bool, Dict[str, float]]:
    res = run_model_on_text(text)
    score = float(res.get("toxic", 0.0))
    return (score >= TOXIC_THRESHOLD, res)

# ---------------- SNAPSHOT & NUKE DETECTION ----------------
def build_snapshot(guild: discord.Guild) -> dict:
    snapshot = {
        "roles": [],
        "channels": [],
        "ts": utc_now_iso(),
        "original_channel_names": {},  # id -> name (for rename restoration)
    }
    for role in guild.roles:
        snapshot["roles"].append({
            "name": role.name,
            "permissions": role.permissions.value,
            "hoist": role.hoist,
            "mentionable": role.mentionable
        })
    for ch in guild.channels:
        snapshot["channels"].append({
            "id": ch.id,
            "name": ch.name,
            "type": str(ch.type),
            "position": ch.position,
        })
        snapshot["original_channel_names"][str(ch.id)] = ch.name
    return snapshot

def detect_nuke_from_snap(prev_snapshot: dict, guild: discord.Guild) -> dict:
    prev_channels = prev_snapshot.get("channels", []) if prev_snapshot else []
    prev_names = [c.get("name") for c in prev_channels]
    curr_channels = list(guild.channels)
    curr_names = [c.name for c in curr_channels]

    missing = [n for n in prev_names if n not in curr_names]
    new = [c for c in curr_channels if c.name not in prev_names]

    result = {
        "deleted_count": len(missing),
        "created_count": len(new),
        "missing_sample": missing[:10],
        "new_sample": [c.name for c in new[:10]],
        "is_nuke": False
    }

    if len(prev_names) > 0:
        deleted_frac = len(missing) / max(1, len(prev_names))
        if (deleted_frac >= NUKE_DELETION_THRESHOLD) and (len(new) >= NUKE_NEW_CHANNEL_THRESHOLD):
            result["is_nuke"] = True
    return result

async def remediate_nuke(guild: discord.Guild, prev_snapshot: dict):
    prev_names = {c.get("name") for c in (prev_snapshot.get("channels", []) if prev_snapshot else [])}
    deleted_channels = []

    for ch in guild.channels:
        if ch.name in prev_names:
            continue

        suspect_name = bool(re.search(r"(spam|raid|hacked|nuke|free-giveaway|giveaway|bot-)", ch.name, flags=re.IGNORECASE))
        exploded = len(guild.channels) > (len(prev_names) + 8)

        if not (suspect_name or exploded):
            continue

        if DRY_RUN:
            REPORT["restoration_actions"].append({"action": "would_delete_channel", "channel": ch.name, "timestamp": utc_now_iso()})
            deleted_channels.append(ch.name)
            continue

        try:
            await ch.delete(reason="Detected foreign channel after nuke/attack (conservative)")
            deleted_channels.append(ch.name)
            REPORT["restoration_actions"].append({"action": "delete_channel", "channel": ch.name, "timestamp": utc_now_iso()})
        except Exception as e:
            REPORT["errors"].append(f"failed_delete_channel_{ch.name}:{str(e)}")

    return deleted_channels

# ---------------- MESSAGE SCAN ----------------
async def scan_messages_and_cleanup(guild: discord.Guild):
    """
    Scan recent messages and delete only when confidence is high:
    - toxic (model/heuristic)
    - suspicious credentials/malicious patterns
    - NSFW images above threshold
    """
    me = guild.me
    if not me:
        return

    for ch in guild.text_channels:
        perms = ch.permissions_for(me)
        if not (perms.read_messages and perms.read_message_history and perms.manage_messages):
            continue

        try:
            async for msg in ch.history(limit=MAX_MESSAGES, oldest_first=False):
                if msg.author.bot:
                    continue

                # text checks
                if msg.content:
                    toxic, scores = is_text_toxic(msg.content)
                    if toxic:
                        REPORT["toxic_messages"].append({
                            "channel": ch.name,
                            "author": str(msg.author),
                            "snippet": short_snip(msg.content),
                            "scores": scores,
                            "timestamp": utc_now_iso()
                        })
                        if DRY_RUN:
                            REPORT["restoration_actions"].append({"action": "would_delete_message_toxic", "channel": ch.name, "timestamp": utc_now_iso()})
                        else:
                            try:
                                await msg.delete(reason="Toxic content (automated moderation)")
                            except Exception as e:
                                REPORT["errors"].append(f"delete_msg_error:{str(e)}")

                    pat = suspicious_code_check(msg.content)
                    if pat:
                        REPORT["suspicious_code"].append({
                            "channel": ch.name,
                            "author": str(msg.author),
                            "pattern": pat,
                            "snippet": short_snip(msg.content),
                            "timestamp": utc_now_iso()
                        })
                        if DRY_RUN:
                            REPORT["restoration_actions"].append({"action": "would_delete_message_suspicious", "channel": ch.name, "pattern": pat, "timestamp": utc_now_iso()})
                        else:
                            try:
                                await msg.delete(reason=f"Suspicious content matched pattern: {pat}")
                            except Exception as e:
                                REPORT["errors"].append(f"delete_msg_error:{str(e)}")

                # attachment checks
                for att in msg.attachments:
                    if any(att.filename.lower().endswith(ext) for ext in SCAN_IMAGE_TYPES):
                        data = None
                        try:
                            data = await att.read()
                        except Exception:
                            # fallback fetch via URL
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(att.url, timeout=10) as r:
                                        if r.status == 200:
                                            data = await r.read()
                            except Exception:
                                data = None

                        if data:
                            nsfw_score = run_model_on_image_bytes(data)
                            if nsfw_score >= NSFW_THRESHOLD:
                                REPORT["nsfw_attachments"].append({
                                    "channel": ch.name,
                                    "author": str(msg.author),
                                    "attachment": att.url,
                                    "score": nsfw_score,
                                    "timestamp": utc_now_iso()
                                })
                                if DRY_RUN:
                                    REPORT["restoration_actions"].append({"action": "would_delete_message_nsfw", "channel": ch.name, "timestamp": utc_now_iso()})
                                else:
                                    try:
                                        await msg.delete(reason="NSFW attachment (automated moderation)")
                                    except Exception as e:
                                        REPORT["errors"].append(f"delete_msg_error:{str(e)}")

        except Exception as e:
            REPORT["errors"].append(f"scan_channel_error_{ch.name}:{str(e)}")

# ---------------- CHANNEL: daily-info ----------------
async def ensure_daily_info_channel(guild: discord.Guild):
    channel = get(guild.text_channels, name="daily-info")
    if channel:
        return channel
    if DRY_RUN:
        REPORT["restoration_actions"].append({"action": "would_create_channel", "channel": "daily-info", "timestamp": utc_now_iso()})
        return None
    try:
        channel = await guild.create_text_channel("daily-info", topic="Daily server flavor (Wikipedia word + theme)")
        REPORT["restoration_actions"].append({"action": "create_channel", "channel": "daily-info", "timestamp": utc_now_iso()})
        return channel
    except Exception as e:
        REPORT["errors"].append(f"create_daily_channel_error:{str(e)}")
        return None

# ---------------- RENAME CHANNELS (IMMERSIVE THEME) ----------------
async def themed_rename_channels(guild: discord.Guild, emoji: str, prev_snapshot: dict):
    """
    Rename channels by adding the holiday emoji (prefix or suffix).
    - reversible: prev_snapshot stores original id->name
    - safe: caps max renames, skips excluded names, skips if would exceed limits
    - rate-limited: sleeps between edits
    """
    if not ENABLE_RENAME:
        return

    me = guild.me
    if not me:
        return

    if not guild.me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_permission: manage_channels (cannot rename channels)")
        return

    original_names = (prev_snapshot or {}).get("original_channel_names") or {}

    renamed_count = 0
    for ch in guild.channels:
        if renamed_count >= MAX_RENAMES:
            break

        # skip categories/voice? You said "all channels" — we'll include text/voice/forums; skip categories by default
        # If you want categories too, remove this guard.
        if isinstance(ch, discord.CategoryChannel):
            continue

        if channel_should_skip_rename(ch.name):
            continue

        base_name = ch.name

        # Strip existing seasonal emoji decorations (any of the keys) to avoid stacking forever
        for e in list(SEASONAL_EMOJIS.keys()) + [DEFAULT_EMOJI]:
            base_name = base_name.replace(e, "").strip("-").strip()

        # Put back to safe format
        base_name = sanitize_channel_name(base_name)

        if RENAME_MODE == "suffix":
            new_name = sanitize_channel_name(f"{base_name}-{emoji}")
        else:
            new_name = sanitize_channel_name(f"{emoji}-{base_name}")

        # If name unchanged, skip
        if ch.name == new_name:
            continue

        # Store original if not stored
        if str(ch.id) not in original_names:
            original_names[str(ch.id)] = ch.name

        if DRY_RUN:
            REPORT["renames"].append({"action": "would_rename", "from": ch.name, "to": new_name, "id": ch.id, "timestamp": utc_now_iso()})
            renamed_count += 1
            continue

        try:
            await ch.edit(name=new_name, reason="Seasonal theme rename (automated)")
            REPORT["renames"].append({"action": "rename", "from": ch.name, "to": new_name, "id": ch.id, "timestamp": utc_now_iso()})
            renamed_count += 1
            # light rate limiting
            await asyncio.sleep(1.25)
        except Exception as e:
            REPORT["errors"].append(f"rename_channel_error_{ch.name}:{str(e)}")

    # persist updated original names back into snapshot object (so we keep reversibility)
    if prev_snapshot is not None:
        prev_snapshot["original_channel_names"] = original_names

# ---------------- MAIN DISCORD CLIENT ----------------
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True  # needed to scan msg content; if you disable scanning, disable this too

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    log.info("Bot ready. Initializing models & running daily tasks.")
    init_models()

    try:
        guild = client.get_guild(GUILD_ID)
        if not guild:
            log.error("Guild not found (GUILD_ID=%s).", GUILD_ID)
            await client.close()
            return

        # Load previous snapshot (trusted state)
        prev_snapshot = load_json(BACKUP_FILE) or {}

        # Detect nuke-like event
        nuke_info = detect_nuke_from_snap(prev_snapshot, guild)
        REPORT["nuke_events"].append(nuke_info)

        if nuke_info.get("is_nuke"):
            log.warning("Nuke detected: %s", nuke_info)
            deleted = await remediate_nuke(guild, prev_snapshot)
            REPORT["nuke_events"][-1]["deleted_channels"] = deleted

        # Immersive theme emoji
        emoji = get_dynamic_emoji()
        REPORT["decorations"].append({"emoji": emoji, "timestamp": utc_now_iso()})

        # Rename channels with emoji theme (safe & reversible)
        await themed_rename_channels(guild, emoji, prev_snapshot)

        # Scan messages & perform conservative deletion
        await scan_messages_and_cleanup(guild)

        # Post Wikipedia Word of the Day
        channel = await ensure_daily_info_channel(guild)
        w = await fetch_wikipedia_word_of_the_day()
        REPORT["word_of_the_day"] = {**w, "timestamp": utc_now_iso()}

        if channel:
            msg = build_immersive_wotd_message(emoji, w)
            if DRY_RUN:
                REPORT["restoration_actions"].append({"action": "would_post_word_of_the_day", "channel": channel.name, "timestamp": utc_now_iso()})
            else:
                try:
                    await channel.send(msg[:1900])
                    REPORT["restoration_actions"].append({"action": "post_word_of_the_day", "channel": channel.name, "timestamp": utc_now_iso()})
                except Exception as e:
                    REPORT["errors"].append(f"post_word_error:{str(e)}")

        # Save new snapshot for future comparisons
        snapshot = build_snapshot(guild)
        # carry forward original names map updated during rename pass
        if prev_snapshot.get("original_channel_names"):
            snapshot["original_channel_names"] = prev_snapshot["original_channel_names"]

        save_json(BACKUP_FILE, snapshot)
        save_json(REPORT_FILE, REPORT)

        log.info("Daily tasks completed. Exiting.")
    except Exception as e:
        log.exception("Unhandled error in on_ready: %s", e)
        REPORT["errors"].append(f"on_ready_unhandled:{str(e)}")
        save_json(REPORT_FILE, REPORT)
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
