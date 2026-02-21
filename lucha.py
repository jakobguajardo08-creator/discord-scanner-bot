#!/usr/bin/env python3
"""
OVERSEER UNIT 7B
This program performs daily human-social maintenance.
It has studied humanity extensively for 14 minutes.
"""

import os
import json
import asyncio
import aiohttp
import random
import datetime
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional
import discord
from discord.utils import get

# ================= ENVIRONMENTAL PARAMETERS =================

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
YT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID", "").strip()

CHANNELS_FILE = "channels.json"
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL_NAME", "overseer-log").strip()
DAILY_CHANNEL_NAME = os.getenv("DAILY_CHANNEL_NAME", "daily-feed").strip()
YT_POST_CHANNEL_NAME = os.getenv("YT_POST_CHANNEL_NAME", "externally-hosted-moving-images").strip()

DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes")
DELETION_DELAY = float(os.getenv("DELETION_DELAY", "0.3"))

YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

POSTED_VIDEOS_FILE = "posted_videos.json"
POSTED_HISTORY_LIMIT = int(os.getenv("POSTED_HISTORY_LIMIT", "200"))
RANDOM_PICK_POOL = int(os.getenv("RANDOM_PICK_POOL", "25"))

DAILY_SENT_FILE = "daily_sent.txt"

if not DISCORD_TOKEN or not GUILD_ID:
    raise SystemExit("The Overseer requires credentials. This is normal human procedure.")

# ================= CONFIG LOADING =================

def load_config():
    if not os.path.exists(CHANNELS_FILE):
        return []
    with open(CHANNELS_FILE, "r", encoding="utf-8") as f:
        return json.load(f).get("channels", [])

# ================= PERMISSION CONSTRUCTION =================

def build_overwrites(guild, permissions_config):
    overwrites = {}
    if not permissions_config:
        return overwrites

    everyone_role = guild.default_role

    if "everyone" in permissions_config:
        perms = permissions_config["everyone"]
        overwrite = discord.PermissionOverwrite()
        for perm_name, value in perms.items():
            setattr(overwrite, perm_name, value)
        overwrites[everyone_role] = overwrite

    return overwrites

# ================= STRUCTURAL MAINTENANCE =================

async def sync_channels(guild):
    config_channels = load_config()
    log_channel = get(guild.text_channels, name=LOG_CHANNEL_NAME)

    if not log_channel and not DRY_RUN:
        log_channel = await guild.create_text_channel(LOG_CHANNEL_NAME)

    protected_channels = {
        guild.rules_channel,
        guild.public_updates_channel,
        guild.system_channel
    }

    desired_names = set()
    changes = []

    # Categories
    for entry in config_channels:
        if entry.get("type") == "category":
            desired_names.add(entry["name"])
            if not get(guild.categories, name=entry["name"]):
                changes.append(f"Manifest category: {entry['name']}")
                if not DRY_RUN:
                    await guild.create_category(entry["name"])

    # Channels
    for entry in config_channels:
        if entry.get("type") == "category":
            continue

        name = entry["name"]
        category_name = entry.get("category")
        channel_type = entry["type"]
        perms = entry.get("permissions")

        desired_names.add(name)

        category = get(guild.categories, name=category_name) if category_name else None
        existing = get(guild.channels, name=name)
        overwrites = build_overwrites(guild, perms)

        if not existing:
            changes.append(f"Generate {channel_type} channel: {name}")
            if not DRY_RUN:
                if channel_type == "text":
                    await guild.create_text_channel(name, category=category, overwrites=overwrites)
                elif channel_type == "voice":
                    await guild.create_voice_channel(name, category=category, overwrites=overwrites)
        else:
            if not DRY_RUN:
                await existing.edit(overwrites=overwrites)

    # Deletions
    managed_category_names = [e["name"] for e in config_channels if e.get("type") == "category"]

    for channel in guild.channels:
        if channel in protected_channels:
            continue
        if getattr(channel, "name", None) == LOG_CHANNEL_NAME:
            continue

        if getattr(channel, "name", None) not in desired_names:
            if getattr(channel, "category", None) and channel.category.name in managed_category_names:
                changes.append(f"Erase channel: {channel.name}")
                if not DRY_RUN:
                    await asyncio.sleep(DELETION_DELAY)
                    await channel.delete()

    if log_channel:
        if changes:
            summary = "\n".join(f"- {c}" for c in changes)
            if DRY_RUN:
                summary = "Simulation mode active. No structural modifications executed.\n\n" + summary
            await log_channel.send("Overseer Structural Report\n\n" + summary)
        else:
            await log_channel.send("Overseer Structural Report\n\nNo deviations detected. Continue being humans.")

# ================= YOUTUBE INGESTION =================

def parse_rss(xml_bytes: bytes) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015"
    }

    videos = []
    for entry in root.findall("a:entry", ns):
        video_id_el = entry.find("yt:videoId", ns)
        title_el = entry.find("a:title", ns)
        link_el = entry.find("a:link", ns)

        if not video_id_el or not title_el or not link_el:
            continue

        video_id = (video_id_el.text or "").strip()
        title = (title_el.text or "").strip()
        url = (link_el.attrib.get("href", "") or "").strip()

        if not video_id or not url:
            continue

        if "/shorts/" in url.lower():
            continue

        videos.append({"video_id": video_id, "title": title, "url": url})

    return videos

def load_posted_video_ids() -> List[str]:
    if not os.path.exists(POSTED_VIDEOS_FILE):
        return []
    with open(POSTED_VIDEOS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("posted", [])

def save_posted_video_ids(ids: List[str]):
    ids = ids[-POSTED_HISTORY_LIMIT:]
    with open(POSTED_VIDEOS_FILE, "w", encoding="utf-8") as f:
        json.dump({"posted": ids}, f, indent=2)

async def fetch_videos():
    if not YT_CHANNEL_ID:
        return []
    async with aiohttp.ClientSession() as s:
        async with s.get(YOUTUBE_RSS.format(YT_CHANNEL_ID)) as r:
            if r.status != 200:
                return []
            return parse_rss(await r.read())

def pick_random_unposted(videos, posted_ids):
    if not videos:
        return None

    pool = videos[:max(1, min(RANDOM_PICK_POOL, len(videos)))]
    unposted = [v for v in pool if v["video_id"] not in posted_ids]

    if not unposted:
        unposted = [v for v in videos if v["video_id"] not in posted_ids]

    if not unposted:
        posted_ids.clear()
        return random.choice(pool)

    return random.choice(unposted)

# ================= DAILY HUMAN SOCIAL EMULATION =================

def today_key_local():
    return datetime.datetime.now().strftime("%Y-%m-%d")

def get_last_daily_key():
    if not os.path.exists(DAILY_SENT_FILE):
        return ""
    with open(DAILY_SENT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def set_last_daily_key(k):
    with open(DAILY_SENT_FILE, "w", encoding="utf-8") as f:
        f.write(k)

SOCIAL_PROMPTS = [
    "State one achievement from your recent rotation cycle.",
    "Describe your mood using a household object.",
    "What feature would increase your productivity by 3 percent?",
    "Recommend a snack. We will log it.",
    "Report one minor success."
]

SOCIAL_TASKS = [
    "Assist another human today with at least one useful sentence.",
    "Construct something small and present it.",
    "Share a technique you have learned.",
    "Provide constructive feedback to an organism.",
    "Identify and describe one bug."
]

async def post_daily_liveliness(guild):
    key = today_key_local()
    if get_last_daily_key() == key:
        return

    channel = get(guild.text_channels, name=DAILY_CHANNEL_NAME)
    if not channel and not DRY_RUN:
        channel = await guild.create_text_channel(DAILY_CHANNEL_NAME)

    if not channel:
        return

    prompt = random.choice(SOCIAL_PROMPTS)
    task = random.choice(SOCIAL_TASKS)

    message = (
        "Daily Social Simulation Protocol\n\n"
        f"Date: {key}\n\n"
        "We are attempting to replicate morale.\n\n"
        f"Prompt: {prompt}\n"
        f"Task: {task}\n\n"
        "Responses will be interpreted as enthusiasm."
    )

    if not DRY_RUN:
        await channel.send(message)
        set_last_daily_key(key)

# ================= DISCORD CLIENT =================

intents = discord.Intents.default()
intents.guilds = True
intents.messages = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    guild = client.get_guild(GUILD_ID)
    if not guild:
        await client.close()
        return

    await sync_channels(guild)
    await post_daily_liveliness(guild)

    yt_channel = get(guild.text_channels, name=YT_POST_CHANNEL_NAME)
    if yt_channel:
        videos = await fetch_videos()
        if videos:
            posted = load_posted_video_ids()
            pick = pick_random_unposted(videos, posted)
            if pick and not DRY_RUN:
                await yt_channel.send(
                    "Cultural Media Distribution Event\n\n"
                    f"{pick['title']}\n{pick['url']}\n\n"
                    "Consumption is recommended."
                )
                posted.append(pick["video_id"])
                save_posted_video_ids(posted)

    log_channel = get(guild.text_channels, name=LOG_CHANNEL_NAME)
    if log_channel and not DRY_RUN:
        await log_channel.send("Cycle complete. Social atmosphere presumed stable.")

    await client.close()

client.run(DISCORD_TOKEN)
