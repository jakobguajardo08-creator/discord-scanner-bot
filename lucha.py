#!/usr/bin/env python3
"""
OVERSEER UNIT 7B (SOCIAL IMITATION BUILD)

Changes in this version:
- YouTube posting is harder to fail silently:
  - Supports either YT_CHANNEL_ID (channel_id=...) OR a full YT_RSS_URL override
  - Adds detailed logging to overseer-log when the feed is missing / empty / HTTP fails
  - Validates it found at least one non-shorts entry before claiming success
- Adds comical, incorrect modern slang usage (no emojis, no real emotions, only emulation)
- Still: sync channels, post daily ritual once/day, post a random non-repeating YouTube video, exit.
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

# Provide one of:
# 1) YT_CHANNEL_ID (preferred): the channel_id string (looks like "UCxxxx...")
# 2) YT_RSS_URL: a full RSS feed url (for maximum certainty)
YT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID", "").strip()
YT_RSS_URL = os.getenv("YT_RSS_URL", "").strip()

CHANNELS_FILE = "channels.json"
LOG_CHANNEL_NAME = os.getenv("LOG_CHANNEL_NAME", "overseer-log").strip()
DAILY_CHANNEL_NAME = os.getenv("DAILY_CHANNEL_NAME", "daily-feed").strip()
YT_POST_CHANNEL_NAME = os.getenv("YT_POST_CHANNEL_NAME", "externally-hosted-moving-images").strip()

DRY_RUN = os.getenv("DRY_RUN", "false").lower() in ("1", "true", "yes")
DELETION_DELAY = float(os.getenv("DELETION_DELAY", "0.3"))

POSTED_VIDEOS_FILE = "posted_videos.json"
POSTED_HISTORY_LIMIT = int(os.getenv("POSTED_HISTORY_LIMIT", "200"))
RANDOM_PICK_POOL = int(os.getenv("RANDOM_PICK_POOL", "25"))

DAILY_SENT_FILE = "daily_sent.txt"

YOUTUBE_RSS_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

if not DISCORD_TOKEN or not GUILD_ID:
    raise SystemExit("The Overseer requires credentials. This is normal human procedure.")

# ================= UTILS =================

def now_key_local() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d")

def safe_json_load(path: str, default):
    try:
        if not os.path.exists(path):
            return default
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

async def log_line(guild: discord.Guild, text: str):
    ch = get(guild.text_channels, name=LOG_CHANNEL_NAME)
    if ch and not DRY_RUN:
        await ch.send(text)

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

    # Deletions (managed categories only)
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

    videos: List[Dict[str, str]] = []
    for entry in root.findall("a:entry", ns):
        video_id_el = entry.find("yt:videoId", ns)
        title_el = entry.find("a:title", ns)
        link_el = entry.find("a:link", ns)

        if video_id_el is None or title_el is None or link_el is None:
            continue

        video_id = (video_id_el.text or "").strip()
        title = (title_el.text or "").strip()
        url = (link_el.attrib.get("href", "") or "").strip()

        if not video_id or not url:
            continue

        # filter shorts
        if "/shorts/" in url.lower():
            continue

        videos.append({"video_id": video_id, "title": title, "url": url})

    return videos

def load_posted_video_ids() -> List[str]:
    data = safe_json_load(POSTED_VIDEOS_FILE, {"posted": []})
    posted = data.get("posted", []) if isinstance(data, dict) else []
    return [str(x) for x in posted]

def save_posted_video_ids(ids: List[str]):
    ids = ids[-POSTED_HISTORY_LIMIT:]
    with open(POSTED_VIDEOS_FILE, "w", encoding="utf-8") as f:
        json.dump({"posted": ids}, f, indent=2)

def get_feed_url() -> str:
    if YT_RSS_URL:
        return YT_RSS_URL
    if YT_CHANNEL_ID:
        return YOUTUBE_RSS_TEMPLATE.format(YT_CHANNEL_ID)
    return ""

async def fetch_videos(guild: discord.Guild) -> List[Dict[str, str]]:
    feed_url = get_feed_url()
    if not feed_url:
        await log_line(guild, "YouTube subsystem: no feed configured. Set YT_CHANNEL_ID or YT_RSS_URL.")
        return []

    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(feed_url, timeout=aiohttp.ClientTimeout(total=20)) as r:
                if r.status != 200:
                    await log_line(guild, f"YouTube subsystem: feed request failed. HTTP {r.status}. URL used: {feed_url}")
                    return []
                raw = await r.read()
    except Exception as e:
        await log_line(guild, f"YouTube subsystem: request exception. {type(e).__name__}: {e}")
        return []

    try:
        vids = parse_rss(raw)
    except Exception as e:
        await log_line(guild, f"YouTube subsystem: parse exception. {type(e).__name__}: {e}")
        return []

    if not vids:
        await log_line(guild, f"YouTube subsystem: feed parsed but yielded zero non-shorts entries. URL used: {feed_url}")
    return vids

def pick_random_unposted(videos: List[Dict[str, str]], posted_ids: List[str]) -> Optional[Dict[str, str]]:
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

def get_last_daily_key() -> str:
    if not os.path.exists(DAILY_SENT_FILE):
        return ""
    with open(DAILY_SENT_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def set_last_daily_key(k: str):
    safe_write_text(DAILY_SENT_FILE, k)

# Incorrect slang injection (mechanical, awkward, misapplied)
ALIEN_SLANG_OPENERS = [
    "Hello humans. Initiating vibe check protocol. This is not emotional. This is computation.",
    "Greetings. I am here to do the thing. Allegedly this is peak. I am saying this because it is trendy.",
    "Attention chat participants. I will now be so real. This statement is a simulation.",
    "Hello. I have been informed this is giving. I do not know what it is giving. Proceeding anyway."
]

ALIEN_SLANG_TAGS = [
    "This is certified. I believe that means approved.",
    "No cap. I do not possess a hat.",
    "Low key high key. Unclear. Logging both.",
    "It is a slay. I am using this word incorrectly on purpose.",
    "We are locked in. Doors were not involved."
]

SOCIAL_PROMPTS = [
    "State one achievement from your recent rotation cycle.",
    "Describe your mood using a household object.",
    "What feature would increase your productivity by 3 percent?",
    "Recommend a snack. It will be archived.",
    "Provide one small success. The system will label it as 'W'."
]

SOCIAL_TASKS = [
    "Assist another human today with at least one useful sentence.",
    "Construct something small and present it for evaluation.",
    "Share a technique you have learned. This will be treated as lore.",
    "Provide constructive feedback to another organism.",
    "Identify and describe one bug. If none exist, describe a theoretical bug."
]

async def post_daily_liveliness(guild: discord.Guild):
    key = now_key_local()
    if get_last_daily_key() == key:
        return

    channel = get(guild.text_channels, name=DAILY_CHANNEL_NAME)
    if not channel and not DRY_RUN:
        channel = await guild.create_text_channel(DAILY_CHANNEL_NAME)
    if not channel:
        return

    opener = random.choice(ALIEN_SLANG_OPENERS)
    tag = random.choice(ALIEN_SLANG_TAGS)
    prompt = random.choice(SOCIAL_PROMPTS)
    task = random.choice(SOCIAL_TASKS)

    message = (
        "Daily Social Simulation Protocol\n\n"
        f"Date: {key}\n\n"
        f"{opener}\n"
        f"Status tag: {tag}\n\n"
        "Directive block follows.\n\n"
        f"Prompt: {prompt}\n"
        f"Task: {task}\n\n"
        "Responses will be interpreted as 'engagement'. This is apparently good. Allegedly."
    )

    if not DRY_RUN:
        await channel.send(message)
        set_last_daily_key(key)

# ================= DISCORD CLIENT =================

intents = discord.Intents.default()
intents.guilds = True
# message_content is NOT required for sending messages, but can be enabled if you later add reading/processing.
# intents.message_content = True

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
    if not yt_channel:
        await log_line(guild, f"YouTube subsystem: target channel not found: {YT_POST_CHANNEL_NAME}")
    else:
        videos = await fetch_videos(guild)
        if videos:
            posted = load_posted_video_ids()
            pick = pick_random_unposted(videos, posted)
            if pick:
                caption = random.choice([
                    "Cultural Media Distribution Event. This is content. It is serving content.",
                    "Uploading moving picture for human enjoyment. This is my era. Allegedly.",
                    "Here is the video drop. I have been told this is a W.",
                    "Deploying audiovisual artifact. Consider it a vibe. No cap."
                ])

                if not DRY_RUN:
                    try:
                        await yt_channel.send(f"{caption}\n\n{pick['title']}\n{pick['url']}")
                        posted.append(pick["video_id"])
                        save_posted_video_ids(posted)
                        await log_line(guild, f"YouTube subsystem: posted video_id={pick['video_id']} from feed pool_size={min(RANDOM_PICK_POOL, len(videos))}")
                    except Exception as e:
                        await log_line(guild, f"YouTube subsystem: failed to send message. {type(e).__name__}: {e}")
            else:
                await log_line(guild, "YouTube subsystem: no pick available after filtering. Feed may be empty.")

    if not DRY_RUN:
        await log_line(guild, "Cycle complete. Social atmosphere presumed stable. This is an imitation statement.")

    await client.close()

client.run(DISCORD_TOKEN)
