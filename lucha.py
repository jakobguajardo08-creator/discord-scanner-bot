#!/usr/bin/env python3
"""
LuCha — Daily Artist Fan + Channel Security Bot

• Runs once per day
• Enforces channels.json (restore missing, delete extras)
• Posts latest YouTube upload (RSS)
• Safe for CI (GitHub Actions / cron)
"""

import os
import sys
import json
import asyncio
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, date
from typing import Dict, Optional, Set

import aiohttp
import discord
from discord.utils import get

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
YT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID", "")
CHANNELS_FILE = "channels.json"
STATE_FILE = "lucha_state.json"
REPORT_FILE = "mod_report.json"

LUCHA_ARMED = os.getenv("LUCHA_ARMED", "0") == "1"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1" or not LUCHA_ARMED

INFO_CHANNEL = "daily-info"
YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

if not DISCORD_TOKEN or not GUILD_ID:
    print("Missing DISCORD_TOKEN or GUILD_ID")
    sys.exit(1)

# ---------------- HELPERS ----------------
def utc_now():
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def today_key():
    return f"ran:{date.today().isoformat()}"

def make_sigil():
    h = hashlib.sha256(date.today().isoformat().encode()).hexdigest()[:8]
    return f"RB-{h}"

# ---------------- YOUTUBE ----------------
def parse_rss(data: bytes) -> Optional[Dict[str, str]]:
    try:
        root = ET.fromstring(data)
        ns = {"a": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
        entry = root.find("a:entry", ns)
        if not entry:
            return None
        return {
            "video_id": entry.find("yt:videoId", ns).text,
            "title": entry.find("a:title", ns).text,
            "url": entry.find("a:link", ns).attrib["href"],
            "published": entry.find("a:published", ns).text,
        }
    except Exception:
        return None

async def fetch_latest_video():
    if not YT_CHANNEL_ID:
        return None
    async with aiohttp.ClientSession() as s:
        async with s.get(YOUTUBE_RSS.format(YT_CHANNEL_ID)) as r:
            if r.status != 200:
                return None
            return parse_rss(await r.read())

# ---------------- DISCORD ----------------
intents = discord.Intents.default()
intents.guilds = True
client = discord.Client(intents=intents)

REPORT = {
    "ts": None,
    "dry_run": DRY_RUN,
    "actions": [],
    "errors": [],
}

@client.event
async def on_ready():
    REPORT["ts"] = utc_now().isoformat()
    state = load_json(STATE_FILE)

    # ---- once-per-day lock ----
    if state.get(today_key()):
        REPORT["actions"].append("exit_already_ran_today")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    guild = client.get_guild(GUILD_ID)
    if not guild:
        REPORT["errors"].append("guild_not_found")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    spec = load_json(CHANNELS_FILE).get("channels", [])

    def key(t, n): return f"{t}:{n.lower()}"
    wanted: Set[str] = {key(c["type"], c["name"]) for c in spec}

    # ---- safety: system channels ----
    system_ids = set()
    for attr in ("system_channel", "rules_channel", "public_updates_channel"):
        ch = getattr(guild, attr, None)
        if ch:
            system_ids.add(ch.id)

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_manage_channels")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- restore missing ----
    for c in spec:
        if not any(ch.name.lower() == c["name"].lower() for ch in guild.channels):
            if DRY_RUN:
                REPORT["actions"].append(f"would_create:{c['name']}")
                continue
            cat = get(guild.categories, name=c.get("category"))
            if c["type"] == "category":
                await guild.create_category(c["name"])
            elif c["type"] == "text":
                await guild.create_text_channel(c["name"], category=cat)
            elif c["type"] == "voice":
                await guild.create_voice_channel(c["name"], category=cat)
            REPORT["actions"].append(f"create:{c['name']}")
            await asyncio.sleep(1.1)

    # ---- delete foreign ----
    for ch in list(guild.channels):
        if ch.id in system_ids:
            continue
        if isinstance(ch, discord.CategoryChannel):
            t = "category"
        elif isinstance(ch, discord.TextChannel):
            t = "text"
        elif isinstance(ch, discord.VoiceChannel):
            t = "voice"
        else:
            continue

        if key(t, ch.name) not in wanted:
            if DRY_RUN:
                REPORT["actions"].append(f"would_delete:{ch.name}")
                continue
            await ch.delete(reason="LuCha daily security scan")
            REPORT["actions"].append(f"delete:{ch.name}")
            await asyncio.sleep(1.1)

    # ---- fan post ----
    info = get(guild.text_channels, name=INFO_CHANNEL)
    latest = await fetch_latest_video()
    sigil = make_sigil()

    if info:
        if latest:
            msg = (
                f"🎧 **Daily Artist Drop** · `{sigil}`\n"
                f"📺 **New upload:** {latest['title']}\n{latest['url']}"
            )
        else:
            msg = (
                f"🎧 **Daily Artist Check-In** · `{sigil}`\n"
                f"No new uploads yet — stay tuned and keep supporting 🔊"
            )

        if DRY_RUN:
            REPORT["actions"].append("would_post_daily_info")
        else:
            await info.send(msg)
            REPORT["actions"].append("post_daily_info")

    # ---- finalize ----
    state[today_key()] = True
    save_json(STATE_FILE, state)
    save_json(REPORT_FILE, REPORT)
    await client.close()

client.run(DISCORD_TOKEN)
