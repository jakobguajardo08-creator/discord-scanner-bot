#!/usr/bin/env python3
import os
import json
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Dict
from bs4 import BeautifulSoup
import discord
from discord.utils import get

# ================= ENV =================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
YT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID", "").strip()

CHANNELS_FILE = "channels.json"
LAST_VIDEO_FILE = "last_video.txt"
LOG_CHANNEL_NAME = "overseer-log"
DRY_RUN = False
DELETION_DELAY = 0.3

YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

if not DISCORD_TOKEN or not GUILD_ID:
    raise SystemExit("Overseer requires credentials.")

# ================= LOAD CONFIG =================
def load_config():
    if not os.path.exists(CHANNELS_FILE):
        return []
    with open(CHANNELS_FILE, "r") as f:
        return json.load(f).get("channels", [])

# ================= PERMISSION BUILDER =================
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

# ================= CHANNEL SYNC =================
async def sync_channels(guild):
    config_channels = load_config()
    log_channel = get(guild.text_channels, name=LOG_CHANNEL_NAME)

    if not log_channel:
        log_channel = await guild.create_text_channel(LOG_CHANNEL_NAME)

    protected_channels = {
        guild.rules_channel,
        guild.public_updates_channel,
        guild.system_channel
    }

    desired_names = set()
    changes = []

    # --- First Pass: Categories ---
    for entry in config_channels:
        if entry["type"] == "category":
            desired_names.add(entry["name"])
            if not get(guild.categories, name=entry["name"]):
                changes.append(f"Create category: {entry['name']}")
                if not DRY_RUN:
                    await guild.create_category(entry["name"])

    # --- Second Pass: Channels ---
    for entry in config_channels:
        if entry["type"] == "category":
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
            changes.append(f"Create {channel_type}: {name}")
            if not DRY_RUN:
                if channel_type == "text":
                    await guild.create_text_channel(name, category=category, overwrites=overwrites)
                elif channel_type == "voice":
                    await guild.create_voice_channel(name, category=category, overwrites=overwrites)
        else:
            # Update permissions if exists
            if not DRY_RUN:
                await existing.edit(overwrites=overwrites)

    # --- Deletion Phase (ONLY delete managed channels not in JSON) ---
    for channel in guild.channels:

        if channel in protected_channels:
            continue

        if channel.name == LOG_CHANNEL_NAME:
            continue

        # Only delete if channel is in a category defined in JSON
        if channel.name not in desired_names:

            # Only delete if channel belongs to a category we manage
            if channel.category and channel.category.name in [
                e["name"] for e in config_channels if e["type"] == "category"
            ]:

                changes.append(f"Delete {channel.name}")
                if not DRY_RUN:
                    await asyncio.sleep(DELETION_DELAY)
                    await channel.delete()

    # --- Report ---
    if changes:
        summary = "\n".join(changes)
        if DRY_RUN:
            summary = "**DRY RUN — No changes applied**\n\n" + summary
        await log_channel.send(f"Overseer Report:\n\n{summary}")
    else:
        await log_channel.send("Overseer Report:\nNo structural deviations detected.")

# ================= YOUTUBE =================
def parse_rss(xml_bytes: bytes) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015"
    }

    videos = []
    for entry in root.findall("a:entry", ns):
        video_id = entry.find("yt:videoId", ns).text
        title = entry.find("a:title", ns).text
        url = entry.find("a:link", ns).attrib["href"]

        if "/shorts/" in url.lower():
            continue

        videos.append({
            "video_id": video_id,
            "title": title,
            "url": url
        })

    return videos


def get_last_video():
    if not os.path.exists(LAST_VIDEO_FILE):
        return None
    with open(LAST_VIDEO_FILE, "r") as f:
        return f.read().strip()


def save_last_video(video_id):
    with open(LAST_VIDEO_FILE, "w") as f:
        f.write(video_id)


async def fetch_videos():
    if not YT_CHANNEL_ID:
        return []
    async with aiohttp.ClientSession() as s:
        async with s.get(YOUTUBE_RSS.format(YT_CHANNEL_ID)) as r:
            if r.status != 200:
                return []
            return parse_rss(await r.read())

# ================= DISCORD =================
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

    yt_channel = get(guild.text_channels, name="externally-hosted-moving-images")
    if yt_channel:
        videos = await fetch_videos()
        if videos:
            newest = videos[0]
            last = get_last_video()

            if newest["video_id"] != last:
                await yt_channel.send(f"{newest['title']}\n{newest['url']}")
                save_last_video(newest["video_id"])

    await client.close()

client.run(DISCORD_TOKEN)
