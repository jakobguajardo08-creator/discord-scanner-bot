#!/usr/bin/env python3
"""
LuCha — Daily Recon Bot (Channel JSON Authority) + ᴚentBoy Fan Daily + YouTube RSS

What this version does:
- CHANNELS_FILE is the source of truth (desired categories/text/voice channels).
- Deletes any channels not in CHANNELS_FILE (except Discord system channels for safety).
- If a nuke is detected, it performs mass restore:
    1) Create missing channels from CHANNELS_FILE
    2) Delete foreign channels not in CHANNELS_FILE
    3) Optionally restore names if backup snapshot includes original names map
- Runs once per day (hard lock via STATE_FILE).
- Posts daily fan message in INFO_CHANNEL_NAME and includes latest ᴚentBoy upload via RSS.

ENV required:
  DISCORD_TOKEN
  GUILD_ID

ENV optional:
  CHANNELS_FILE              (default: channels.json)  # authoritative channel spec
  BACKUP_FILE                (default: guild_backup.json)  # snapshot storage (original names)
  REPORT_FILE                (default: mod_report.json)
  STATE_FILE                 (default: lucha_state.json)

  LUCHA_ARMED=0/1            (default 0) -> if 0 forces DRY_RUN=1
  DRY_RUN=0/1                (default 0)

  ENABLE_ENFORCEMENT=0/1     (default 1)  # delete foreign channels + create missing
  ENABLE_NUKE_RESTORE=0/1    (default 1)
  ENABLE_DAILY_INFO=0/1      (default 1)
  ENABLE_YOUTUBE=0/1         (default 1)

  INFO_CHANNEL_NAME          (default: daily-info)

YouTube RSS:
  YT_CHANNEL_ID              (default: UCtEF5V5tssiKldIAIkrjb0w)  # ᴚentBoy handle @RentBoy-e5h
  YT_POST_TEMPLATE           (default: "📺 **New ᴚentBoy upload:** {title}\n{url}")

Nuke detection (conservative):
  NUKE_DELETION_THRESHOLD    (default 0.20)  # fraction of known channels missing to trigger restore
  NUKE_NEW_CHANNEL_THRESHOLD (default 4)     # number of new channels not in spec to trigger restore

Permissions required (for real runs):
  - Manage Channels (create/delete/edit)
  - Send Messages (for daily info)
"""

from __future__ import annotations

import os
import sys
import json
import re
import asyncio
import logging
import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Optional, List, Any, Tuple, Set

import aiohttp
import discord
from discord.utils import get

# Optional local .env
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

LUCHA_NAME = "LuCha"
YOUTUBE_RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

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
        log.error("save_json failed %s: %s", path, e)

def load_json(path: str) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
    except Exception as e:
        log.error("load_json failed %s: %s", path, e)
    return {}

def redact_for_logs(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"(mfa\.[A-Za-z0-9_\-]{20,})", "mfa.[REDACTED]", s)
    s = re.sub(r"(discord(app)?\.com\/api\/webhooks\/\d+\/)[A-Za-z0-9_\-]+", r"\1[REDACTED]", s)
    s = re.sub(r"([A-Za-z0-9_\-]{23,28}\.[A-Za-z0-9_\-]{6,7}\.[A-Za-z0-9_\-]{10,})", "[REDACTED_TOKEN]", s)
    return s

def normalize_name(name: str) -> str:
    return (name or "").strip().lower()

# ---------------- Config ----------------
@dataclass
class Config:
    discord_token: str
    guild_id: int

    channels_file: str = "channels.json"        # authoritative
    backup_file: str = "guild_backup.json"      # snapshot (for original names map)
    report_file: str = "mod_report.json"
    state_file: str = "lucha_state.json"

    lucha_armed: bool = False
    dry_run: bool = True

    enable_enforcement: bool = True
    enable_nuke_restore: bool = True
    enable_daily_info: bool = True
    enable_youtube: bool = True

    info_channel_name: str = "daily-info"

    yt_channel_id: str = "UCtEF5V5tssiKldIAIkrjb0w"
    yt_post_template: str = "📺 **New ᴚentBoy upload:** {title}\n{url}"

    nuke_deletion_threshold: float = 0.20
    nuke_new_channel_threshold: int = 4

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
            channels_file=env_str("CHANNELS_FILE", "channels.json"),
            backup_file=env_str("BACKUP_FILE", "guild_backup.json"),
            report_file=env_str("REPORT_FILE", "mod_report.json"),
            state_file=env_str("STATE_FILE", "lucha_state.json"),

            lucha_armed=env_bool("LUCHA_ARMED", False),
            dry_run=env_bool("DRY_RUN", False),

            enable_enforcement=env_bool("ENABLE_ENFORCEMENT", True),
            enable_nuke_restore=env_bool("ENABLE_NUKE_RESTORE", True),
            enable_daily_info=env_bool("ENABLE_DAILY_INFO", True),
            enable_youtube=env_bool("ENABLE_YOUTUBE", True),

            info_channel_name=env_str("INFO_CHANNEL_NAME", "daily-info"),

            yt_channel_id=env_str("YT_CHANNEL_ID", "UCtEF5V5tssiKldIAIkrjb0w"),
            yt_post_template=env_str("YT_POST_TEMPLATE", "📺 **New ᴚentBoy upload:** {title}\n{url}"),

            nuke_deletion_threshold=env_float("NUKE_DELETION_THRESHOLD", 0.20, min_v=0.0, max_v=1.0),
            nuke_new_channel_threshold=env_int("NUKE_NEW_CHANNEL_THRESHOLD", 4, min_v=0, max_v=9999),
        )

        # Hard safety: must explicitly arm for destructive edits/posts/deletes
        if not cfg.lucha_armed:
            cfg.dry_run = True

        return cfg

CFG = Config.load()

# ---------------- Report ----------------
REPORT: Dict[str, Any] = {
    "lucha": {
        "name": LUCHA_NAME,
        "armed": CFG.lucha_armed,
        "dry_run": CFG.dry_run,
        "ts": None,
        "version": "rentboy-yt-json-authority-1.0",
    },
    "actions": [],
    "nuke_events": [],
    "daily_info": {},
    "youtube": {},
    "errors": [],
}

# ---------------- Once-per-day lock ----------------
def load_state() -> dict:
    st = load_json(CFG.state_file) or {}
    return st if isinstance(st, dict) else {}

def save_state(st: dict):
    save_json(CFG.state_file, st)

def state_key_today(prefix: str) -> str:
    return f"{prefix}:{today_utc_date().isoformat()}"

def already_ran_today(st: dict) -> bool:
    return st.get(state_key_today("full_run_done")) is True

def mark_ran_today(st: dict):
    st[state_key_today("full_run_done")] = True

# ---------------- Channel spec parsing ----------------
SUPPORTED_TYPES = {"category", "text", "voice"}

def load_channel_spec() -> dict:
    spec = load_json(CFG.channels_file) or {}
    if not spec.get("channels") or not isinstance(spec.get("channels"), list):
        REPORT["errors"].append(f"channels_file_invalid_or_empty:{CFG.channels_file}")
        return {"channels": []}
    return spec

def spec_key(ch_type: str, name: str) -> str:
    return f"{ch_type}:{normalize_name(name)}"

def build_allowed_keys(spec: dict) -> Set[str]:
    keys: Set[str] = set()
    for item in spec.get("channels", []):
        if not isinstance(item, dict):
            continue
        t = normalize_name(str(item.get("type", "")))
        n = normalize_name(str(item.get("name", "")))
        if t in SUPPORTED_TYPES and n:
            keys.add(spec_key(t, n))
    return keys

def get_system_channel_ids(guild: discord.Guild) -> Set[int]:
    """Safety: don't delete Discord-configured system channels even if spec forgot them."""
    ids: Set[int] = set()
    for attr in ("system_channel", "rules_channel", "public_updates_channel", "safety_alerts_channel"):
        ch = getattr(guild, attr, None)
        if ch is not None and getattr(ch, "id", None):
            ids.add(int(ch.id))
    return ids

def guild_channel_key(ch: discord.abc.GuildChannel) -> Optional[str]:
    name = getattr(ch, "name", None)
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(ch, discord.CategoryChannel):
        return spec_key("category", name)
    if isinstance(ch, discord.TextChannel):
        return spec_key("text", name)
    if isinstance(ch, discord.VoiceChannel):
        return spec_key("voice", name)
    return None

# ---------------- Snapshot (for original name restore) ----------------
def build_snapshot(guild: discord.Guild) -> dict:
    snap = {
        "ts": utc_now_iso(),
        "guild_id": guild.id,
        "original_channel_names": {},
    }
    for ch in guild.channels:
        nm = getattr(ch, "name", "")
        snap["original_channel_names"][str(ch.id)] = nm
    return snap

async def restore_channel_names_from_snapshot(guild: discord.Guild, snapshot: dict):
    originals = (snapshot or {}).get("original_channel_names", {}) or {}
    if not originals:
        return

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_manage_channels_for_restore_names")
        return

    for ch in guild.channels:
        target = originals.get(str(ch.id))
        if not target or getattr(ch, "name", "") == target:
            continue

        if CFG.dry_run:
            REPORT["actions"].append({"action": "would_restore_name", "id": ch.id, "to": target, "ts": utc_now_iso()})
            continue

        try:
            await ch.edit(name=target, reason=f"{LUCHA_NAME}: restore names")
            REPORT["actions"].append({"action": "restore_name", "id": ch.id, "to": target, "ts": utc_now_iso()})
            await asyncio.sleep(1.1)
        except Exception as e:
            REPORT["errors"].append(f"restore_name_error:{redact_for_logs(str(e))}")

# ---------------- Mass reconcile (create missing + delete foreign) ----------------
async def ensure_category(guild: discord.Guild, name: str) -> Optional[discord.CategoryChannel]:
    cat = get(guild.categories, name=name)
    if cat:
        return cat

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append(f"missing_manage_channels_create_category:{name}")
        return None

    if CFG.dry_run:
        REPORT["actions"].append({"action": "would_create_category", "name": name, "ts": utc_now_iso()})
        return None

    try:
        cat = await guild.create_category(name=name, reason=f"{LUCHA_NAME}: reconcile category")
        REPORT["actions"].append({"action": "create_category", "name": name, "ts": utc_now_iso()})
        await asyncio.sleep(1.1)
        return cat
    except Exception as e:
        REPORT["errors"].append(f"create_category_error:{name}:{redact_for_logs(str(e))}")
        return None

async def create_missing_channels_from_spec(guild: discord.Guild, spec: dict):
    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_manage_channels_for_create_missing")
        return

    allowed = build_allowed_keys(spec)
    existing_keys = set(k for ch in guild.channels for k in [guild_channel_key(ch)] if k)

    for item in spec.get("channels", []):
        if not isinstance(item, dict):
            continue
        t = normalize_name(str(item.get("type", "")))
        name = str(item.get("name", "")).strip()
        if t not in SUPPORTED_TYPES or not name:
            continue

        key = spec_key(t, name)
        if key in existing_keys:
            continue

        category_name = str(item.get("category", "")).strip()
        topic = str(item.get("topic", "")).strip()

        cat_obj = None
        if category_name and t != "category":
            cat_obj = await ensure_category(guild, category_name)

        if CFG.dry_run:
            REPORT["actions"].append({"action": "would_create_channel", "type": t, "name": name, "category": category_name, "ts": utc_now_iso()})
            continue

        try:
            if t == "category":
                await ensure_category(guild, name)
            elif t == "text":
                await guild.create_text_channel(name=name, category=cat_obj, topic=(topic[:1000] if topic else None),
                                               reason=f"{LUCHA_NAME}: reconcile text channel")
                REPORT["actions"].append({"action": "create_text_channel", "name": name, "category": category_name, "ts": utc_now_iso()})
            elif t == "voice":
                await guild.create_voice_channel(name=name, category=cat_obj, reason=f"{LUCHA_NAME}: reconcile voice channel")
                REPORT["actions"].append({"action": "create_voice_channel", "name": name, "category": category_name, "ts": utc_now_iso()})
            await asyncio.sleep(1.2)
        except Exception as e:
            REPORT["errors"].append(f"create_channel_error_{t}:{name}:{redact_for_logs(str(e))}")

async def delete_foreign_channels_not_in_spec(guild: discord.Guild, spec: dict):
    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_manage_channels_for_delete_foreign")
        return

    allowed = build_allowed_keys(spec)
    system_ids = get_system_channel_ids(guild)

    for ch in list(guild.channels):
        # never delete system-configured channels (extra safety)
        if ch.id in system_ids:
            continue

        key = guild_channel_key(ch)
        if key and key in allowed:
            continue

        # Allow threads etc by ignoring unknown types (only delete category/text/voice)
        if not isinstance(ch, (discord.CategoryChannel, discord.TextChannel, discord.VoiceChannel)):
            continue

        if CFG.dry_run:
            REPORT["actions"].append({"action": "would_delete_foreign_channel", "name": getattr(ch, "name", ""), "id": ch.id, "ts": utc_now_iso()})
            continue

        try:
            await ch.delete(reason=f"{LUCHA_NAME}: not in channel spec")
            REPORT["actions"].append({"action": "delete_foreign_channel", "name": getattr(ch, "name", ""), "id": ch.id, "ts": utc_now_iso()})
            await asyncio.sleep(1.2)
        except Exception as e:
            REPORT["errors"].append(f"delete_foreign_channel_error:{getattr(ch,'name','')}:{redact_for_logs(str(e))}")

async def mass_reconcile(guild: discord.Guild, spec: dict, snapshot_for_names: dict):
    """Create missing + delete foreign + restore names (best-effort)."""
    if not CFG.enable_enforcement:
        return
    await create_missing_channels_from_spec(guild, spec)
    await delete_foreign_channels_not_in_spec(guild, spec)
    await restore_channel_names_from_snapshot(guild, snapshot_for_names)

# ---------------- Nuke detection ----------------
def detect_nuke(guild: discord.Guild, spec: dict) -> dict:
    allowed = build_allowed_keys(spec)
    curr_keys = set(k for ch in guild.channels for k in [guild_channel_key(ch)] if k)

    # missing = spec channels not present
    missing = list(allowed - curr_keys)
    missing_frac = len(missing) / max(1, len(allowed))

    # new/foreign = existing channels that are category/text/voice but not in spec
    system_ids = get_system_channel_ids(guild)
    foreign = []
    for ch in guild.channels:
        if ch.id in system_ids:
            continue
        k = guild_channel_key(ch)
        if k and k not in allowed:
            foreign.append(k)

    is_nuke = (missing_frac >= CFG.nuke_deletion_threshold) or (len(foreign) >= CFG.nuke_new_channel_threshold)

    return {
        "is_nuke": is_nuke,
        "missing_frac": round(missing_frac, 4),
        "missing_count": len(missing),
        "foreign_count": len(foreign),
        "missing_sample": missing[:8],
        "foreign_sample": foreign[:8],
    }

# ---------------- YouTube RSS ----------------
def parse_youtube_rss(xml_bytes: bytes) -> Optional[Dict[str, str]]:
    """Returns latest entry: {video_id, title, url, published}"""
    try:
        root = ET.fromstring(xml_bytes)
        ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None

        title_el = entry.find("atom:title", ns)
        vid_el = entry.find("yt:videoId", ns)
        link_el = entry.find("atom:link", ns)
        pub_el = entry.find("atom:published", ns)

        video_id = (vid_el.text or "").strip() if vid_el is not None else ""
        title = (title_el.text or "").strip() if title_el is not None else "(untitled)"
        url = (link_el.attrib.get("href") or "").strip() if link_el is not None else ""
        published = (pub_el.text or "").strip() if pub_el is not None else ""

        if not video_id or not url:
            return None
        return {"video_id": video_id, "title": title, "url": url, "published": published}
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

# ---------------- Posting ----------------
async def ensure_text_channel(guild: discord.Guild, name: str, topic: str = "") -> Optional[discord.TextChannel]:
    ch = get(guild.text_channels, name=name)
    if ch:
        return ch

    if CFG.dry_run:
        REPORT["actions"].append({"action": "would_create_text_channel_for_posting", "name": name, "ts": utc_now_iso()})
        return None

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append(f"missing_manage_channels_create_text:{name}")
        return None

    try:
        ch = await guild.create_text_channel(name=name, topic=(topic[:1000] if topic else None), reason=f"{LUCHA_NAME}: ensure post channel")
        REPORT["actions"].append({"action": "create_text_channel_for_posting", "name": name, "ts": utc_now_iso()})
        await asyncio.sleep(1.1)
        return ch
    except Exception as e:
        REPORT["errors"].append(f"create_text_channel_error:{name}:{redact_for_logs(str(e))}")
        return None

async def safe_send(channel: discord.TextChannel, content: str, tag: str):
    if CFG.dry_run:
        REPORT["actions"].append({"action": f"would_{tag}", "channel": channel.name, "ts": utc_now_iso()})
        return
    try:
        await channel.send(content[:1900])
        REPORT["actions"].append({"action": tag, "channel": channel.name, "ts": utc_now_iso()})
    except Exception as e:
        REPORT["errors"].append(f"send_error_{tag}:{redact_for_logs(str(e))}")

def make_sigil(guild_id: int) -> str:
    day = today_utc_date().isoformat()
    h = hashlib.sha256(f"{guild_id}:{day}:rentboy:Lucha".encode("utf-8")).hexdigest()[:10]
    return f"RB-{h}"

def build_fan_daily(latest: Optional[Dict[str, str]]) -> str:
    # We only claim what we can observe: channel ethos + latest upload metadata (from RSS).
    sigil = make_sigil(CFG.guild_id)
    ethos = "🧼 **Ethos:** NO generative AI is used in any music or art pieces!!"
    header = f"🎧 **ᴚentBoy Daily** · `{sigil}`"

    if latest:
        drop = f"📺 **Latest upload:** {latest['title']}\n{latest['url']}"
        return f"{header}\n{ethos}\n{drop}"
    return f"{header}\n{ethos}\n(No RSS drop found today — check the channel feed soon.)"

async def post_daily_info_and_latest(guild: discord.Guild, st: dict):
    if not CFG.enable_daily_info:
        return

    # de-dupe daily post
    daily_key = state_key_today("daily_posted")
    if st.get(daily_key) is True:
        REPORT["daily_info"] = {"posted": False, "reason": "already_posted_today", "ts": utc_now_iso()}
        return

    info_ch = await ensure_text_channel(guild, CFG.info_channel_name, topic="Daily ᴚentBoy fan drops + latest uploads (safe automation).")
    if not info_ch:
        REPORT["daily_info"] = {"posted": False, "reason": "no_channel", "ts": utc_now_iso()}
        return

    latest = None
    if CFG.enable_youtube and CFG.yt_channel_id:
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": f"{LUCHA_NAME}/1.0"}) as session:
            latest = await fetch_latest_youtube_video(session, CFG.yt_channel_id)

        # optional: post "new upload" only if new
        last_id = (st.get("yt_last_video_id") or "").strip()
        if latest and latest["video_id"] and latest["video_id"] != last_id:
            yt_msg = CFG.yt_post_template.format(title=latest["title"], url=latest["url"])
            await safe_send(info_ch, yt_msg, "post_youtube_latest")
            if not CFG.dry_run:
                st["yt_last_video_id"] = latest["video_id"]
                st["yt_last_video_url"] = latest["url"]
                st["yt_last_video_published"] = latest.get("published", "")
                st["yt_last_posted_at"] = utc_now_iso()

            REPORT["youtube"] = {"checked": True, "posted": not CFG.dry_run, "video_id": latest["video_id"], "ts": utc_now_iso()}
        else:
            REPORT["youtube"] = {"checked": True, "posted": False, "reason": "same_as_last_or_none", "ts": utc_now_iso()}

    # always post the daily fan message (once/day)
    daily_msg = build_fan_daily(latest)
    await safe_send(info_ch, daily_msg, "post_daily_info")

    st[daily_key] = True
    REPORT["daily_info"] = {"posted": not CFG.dry_run, "channel": info_ch.name, "date": today_utc_date().isoformat(), "ts": utc_now_iso()}

# ---------------- Discord client ----------------
intents = discord.Intents.default()
intents.guilds = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    REPORT["lucha"]["ts"] = utc_now_iso()

    st = load_state()
    if already_ran_today(st):
        REPORT["actions"].append({"action": "exit_already_ran_today", "ts": utc_now_iso()})
        save_json(CFG.report_file, REPORT)
        await client.close()
        return

    try:
        guild = client.get_guild(CFG.guild_id)
        if not guild:
            REPORT["errors"].append(f"guild_not_found:{CFG.guild_id}")
            save_json(CFG.report_file, REPORT)
            await client.close()
            return

        spec = load_channel_spec()
        prev_snapshot = load_json(CFG.backup_file) or {}

        # Detect nuke against the *spec*, not against last snapshot IDs
        nuke_info = detect_nuke(guild, spec)
        REPORT["nuke_events"].append({**nuke_info, "ts": utc_now_iso()})

        if CFG.enable_nuke_restore and nuke_info.get("is_nuke"):
            log.warning("%s detected possible nuke -> mass reconcile", LUCHA_NAME)
            await mass_reconcile(guild, spec, prev_snapshot)
        else:
            # normal daily enforcement too (keeps server clean)
            if CFG.enable_enforcement:
                await mass_reconcile(guild, spec, prev_snapshot)

        # Daily fan post + latest upload
        await post_daily_info_and_latest(guild, st)

        # Save a fresh snapshot for future name restores
        new_snapshot = build_snapshot(guild)
        # preserve original map if we already had one (best-effort)
        if prev_snapshot.get("original_channel_names"):
            new_snapshot["original_channel_names"] = prev_snapshot["original_channel_names"]
        save_json(CFG.backup_file, new_snapshot)

        # mark run complete (hard once/day lock)
        mark_ran_today(st)
        save_state(st)

        save_json(CFG.report_file, REPORT)
        log.info("%s done. armed=%s dry_run=%s report=%s", LUCHA_NAME, CFG.lucha_armed, CFG.dry_run, CFG.report_file)

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
    # Never print the token
    client.run(CFG.discord_token)
