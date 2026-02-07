#!/usr/bin/env python3
import os
import json
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, date
from typing import Optional, Dict, List

import discord
from discord.utils import get

# ===================== CONFIG =====================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GUILD_ID = int(os.getenv("GUILD_ID", "0"))

YT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID", "").strip()

LOCALAI_BASE_URL = os.getenv("LOCALAI_BASE_URL", "").rstrip("/")
LOCALAI_API_KEY = os.getenv("LOCALAI_API_KEY", "local")
LOCALAI_MODEL = os.getenv("LOCALAI_MODEL", "mistral")

STATE_FILE = "lucha_state.json"

SUMMARY_SOURCE_CHANNEL = "unstructured-communication"
SUMMARY_TARGET_CHANNEL = "statements-for-all-individuals"
YT_TARGET_CHANNEL = "externally-hosted-moving-images"

YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

# ===================== SAFETY =====================
if not DISCORD_TOKEN or not GUILD_ID:
    raise SystemExit("Missing DISCORD_TOKEN or GUILD_ID")

# ===================== UTILS =====================
def utc_today_key() -> str:
    return f"daily:{date.today().isoformat()}"

def load_state() -> dict:
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

# ===================== LOCALAI =====================
async def summarize_with_localai(text: str) -> Optional[str]:
    payload = {
        "model": LOCALAI_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Summarize observed human communication in a neutral, "
                    "formal, slightly uncanny tone. "
                    "Do not use humor, emojis, or opinions."
                )
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.4
    }

    headers = {
        "Authorization": f"Bearer {LOCALAI_API_KEY}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as session:
        async with session.post(
            f"{LOCALAI_BASE_URL}/chat/completions",
            json=payload,
            headers=headers
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

# ===================== YOUTUBE =====================
def parse_rss(xml_bytes: bytes) -> Optional[Dict[str, str]]:
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
        "url": entry.find("a:link", ns).attrib["href"]
    }

async def fetch_latest_video() -> Optional[Dict[str, str]]:
    if not YT_CHANNEL_ID:
        return None
    async with aiohttp.ClientSession() as session:
        async with session.get(YOUTUBE_RSS.format(YT_CHANNEL_ID)) as resp:
            if resp.status != 200:
                return None
            return parse_rss(await resp.read())

# ===================== DISCORD =====================
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    state = load_state()
    today = utc_today_key()

    if state.get(today):
        print("Daily run already completed")
        await client.close()
        return

    guild = client.get_guild(GUILD_ID)
    if not guild:
        await client.close()
        return

    source = get(guild.text_channels, name=SUMMARY_SOURCE_CHANNEL)
    target = get(guild.text_channels, name=SUMMARY_TARGET_CHANNEL)
    yt_channel = get(guild.text_channels, name=YT_TARGET_CHANNEL)

    # -------- DAILY SUMMARY (ALWAYS) --------
    if source and target:
        messages: List[str] = []
        async for msg in source.history(limit=75):
            if not msg.author.bot and msg.content:
                messages.append(msg.content)

        if messages:
            summary = await summarize_with_localai("\n".join(messages))
            if summary:
                await target.send(
                    "Daily observed communication summary:\n\n" + summary
                )

    # -------- YOUTUBE CHECK (CONDITIONAL POST) --------
    yt = await fetch_latest_video()
    if yt and yt_channel:
        if yt["video_id"] != state.get("yt_last_video_id"):
            await yt_channel.send(
                f"Externally hosted moving image detected:\n"
                f"{yt['title']}\n{yt['url']}"
            )
            state["yt_last_video_id"] = yt["video_id"]

    # -------- FINALIZE --------
    state[today] = True
    save_state(state)
    print("Daily run complete")
    await client.close()

client.run(DISCORD_TOKEN)
