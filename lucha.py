#!/usr/bin/env python3
"""
LuCha — SAFE Daily Channel Reconciler + YouTube Poster

• channels.json is the source of truth
• Creates missing channels
• Applies permissions
• Deletes foreign channels
• Posts latest YouTube upload (RSS, no API key)
• Runs once per day unless FORCE_RUN=1
"""

import os
import sys
import json
import re
import asyncio
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, date
from typing import Dict, Set, Optional

import aiohttp
import discord
from discord.utils import get

# ================= CONFIG =================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GUILD_ID_RAW = os.getenv("GUILD_ID", "0").strip()
YT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID", "").strip()

CHANNELS_FILE = "channels.json"
STATE_FILE = "lucha_state.json"
REPORT_FILE = "mod_report.json"

LUCHA_ARMED = os.getenv("LUCHA_ARMED", "0") == "1"
FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1" or not LUCHA_ARMED

YT_POST_CHANNEL = "yt-posts"
YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

if not DISCORD_TOKEN or not GUILD_ID_RAW.isdigit():
    print("❌ Missing DISCORD_TOKEN or invalid GUILD_ID")
    sys.exit(1)

GUILD_ID = int(GUILD_ID_RAW)

# ================= HELPERS =================
def utc_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def save_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def today_key() -> str:
    return f"ran:{date.today().isoformat()}"

def sanitize_text_channel_name(name: str) -> str:
    n = (name or "").strip().lower()
    n = n.replace(" ", "-")
    n = re.sub(r"[^a-z0-9\-]", "", n)
    n = re.sub(r"\-+", "-", n).strip("-")
    return n[:100] or "channel"

def type_key(t: str, name: str) -> str:
    return f"{t}:{name.lower()}"

# ================= REPORT =================
REPORT = {
    "timestamp": None,
    "dry_run": DRY_RUN,
    "armed": LUCHA_ARMED,
    "actions": [],
    "errors": [],
    "youtube": {}
}

# ================= PERMISSIONS =================
PERM_MAP = {
    "view_channel": "view_channel",
    "send_messages": "send_messages",
    "add_reactions": "add_reactions",
    "manage_messages": "manage_messages",
    "connect": "connect",
    "speak": "speak",
}

def build_overwrites(guild, perm_spec):
    overwrites = {}
    everyone = guild.default_role

    if isinstance(perm_spec, dict) and "everyone" in perm_spec:
        p = discord.PermissionOverwrite()
        for k, v in perm_spec["everyone"].items():
            if k in PERM_MAP:
                setattr(p, PERM_MAP[k], bool(v))
        overwrites[everyone] = p

    for role_name, rules in perm_spec.get("roles", {}).items() if isinstance(perm_spec, dict) else []:
        role = get(guild.roles, name=role_name)
        if not role:
            continue
        p = discord.PermissionOverwrite()
        for k, v in rules.items():
            if k in PERM_MAP:
                setattr(p, PERM_MAP[k], bool(v))
        overwrites[role] = p

    return overwrites

# ================= YOUTUBE =================
def parse_rss(xml_bytes: bytes) -> Optional[Dict[str, str]]:
    try:
        root = ET.fromstring(xml_bytes)
        ns = {
            "a": "http://www.w3.org/2005/Atom",
            "yt": "http://www.youtube.com/xml/schemas/2015"
        }
        entry = root.find("a:entry", ns)
        if entry is None:
            return None
        return {
            "video_id": entry.find("yt:videoId", ns).text,
            "title": entry.find("a:title", ns).text,
            "url": entry.find("a:link", ns).attrib["href"],
            "published": entry.find("a:published", ns).text,
        }
    except Exception:
        return None

async def fetch_latest_video() -> Optional[Dict[str, str]]:
    if not YT_CHANNEL_ID:
        return None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as s:
        async with s.get(YOUTUBE_RSS.format(YT_CHANNEL_ID)) as r:
            if r.status != 200:
                return None
            return parse_rss(await r.read())

# ================= DISCORD =================
intents = discord.Intents.default()
intents.guilds = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    REPORT["timestamp"] = utc_iso()

    # ---- DAILY LOCK ----
    state = load_json(STATE_FILE)
    if state.get(today_key()) and not FORCE_RUN:
        REPORT["actions"].append("exit_already_ran_today")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- LOAD & VALIDATE SPEC ----
    spec_data = load_json(CHANNELS_FILE)
    spec = spec_data.get("channels")

    if not isinstance(spec, list) or len(spec) == 0:
        REPORT["errors"].append("ABORTED: channels.json missing or empty")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    guild = client.get_guild(GUILD_ID)
    if not guild:
        REPORT["errors"].append("guild_not_found")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_manage_channels_permission")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- NORMALIZE SPEC ----
    normalized = []
    for c in spec:
        t = c.get("type")
        name = c.get("name")
        if t == "text":
            name = sanitize_text_channel_name(name)
        normalized.append({**c, "type": t, "name": name})

    allowed = {type_key(c["type"], c["name"]) for c in normalized}

    # ---- CREATE CATEGORIES ----
    category_map = {}
    creation_failed = False

    for c in normalized:
        if c["type"] != "category":
            continue
        cat = get(guild.categories, name=c["name"])
        if cat:
            category_map[c["name"]] = cat
            continue
        if DRY_RUN:
            REPORT["actions"].append(f"would_create_category:{c['name']}")
            continue
        try:
            cat = await guild.create_category(c["name"])
            category_map[c["name"]] = cat
            REPORT["actions"].append(f"create_category:{c['name']}")
            await asyncio.sleep(1.2)
        except Exception as e:
            creation_failed = True
            REPORT["errors"].append(f"category_create_error:{c['name']}:{e}")

    # ---- CREATE CHANNELS ----
    for c in normalized:
        if c["type"] == "category":
            continue
        if get(
            guild.text_channels if c["type"] == "text" else guild.voice_channels,
            name=c["name"]
        ):
            continue

        parent = category_map.get(c.get("category"))
        overwrites = build_overwrites(guild, c.get("permissions", {}))

        if DRY_RUN:
            REPORT["actions"].append(f"would_create_{c['type']}:{c['name']}")
            continue
        try:
            if c["type"] == "text":
                await guild.create_text_channel(c["name"], category=parent, overwrites=overwrites)
            else:
                await guild.create_voice_channel(c["name"], category=parent, overwrites=overwrites)
            REPORT["actions"].append(f"create_{c['type']}:{c['name']}")
            await asyncio.sleep(1.2)
        except Exception as e:
            creation_failed = True
            REPORT["errors"].append(f"channel_create_error:{c['name']}:{e}")

    if creation_failed:
        REPORT["errors"].append("ABORTED: creation failures, skipping deletes")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- DELETE FOREIGN CHANNELS ----
    for ch in list(guild.channels):
        if isinstance(ch, discord.CategoryChannel):
            k = type_key("category", ch.name)
        elif isinstance(ch, discord.TextChannel):
            k = type_key("text", ch.name)
        elif isinstance(ch, discord.VoiceChannel):
            k = type_key("voice", ch.name)
        else:
            continue

        if k in allowed:
            continue

        if DRY_RUN:
            REPORT["actions"].append(f"would_delete:{k}")
            continue
        try:
            await ch.delete(reason="LuCha reconcile")
            REPORT["actions"].append(f"delete:{k}")
            await asyncio.sleep(1.2)
        except Exception as e:
            REPORT["errors"].append(f"delete_error:{k}:{e}")

    # ---- YOUTUBE POSTING ----
    yt = await fetch_latest_video()
    if yt:
        last_id = state.get("yt_last_video_id")
        REPORT["youtube"]["latest"] = yt["video_id"]

        if yt["video_id"] != last_id:
            ch = get(guild.text_channels, name=YT_POST_CHANNEL)
            if ch:
                msg = f"📺 **New upload**\n**{yt['title']}**\n{yt['url']}"
                if DRY_RUN:
                    REPORT["actions"].append("would_post_youtube")
                else:
                    await ch.send(msg)
                    REPORT["actions"].append("post_youtube")
                    state["yt_last_video_id"] = yt["video_id"]
                    state["yt_last_video_url"] = yt["url"]
                    state["yt_last_video_published"] = yt["published"]

    # ---- FINALIZE ----
    state[today_key()] = True
    save_json(STATE_FILE, state)
    save_json(REPORT_FILE, REPORT)
    await client.close()

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
