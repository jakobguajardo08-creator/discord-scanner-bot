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
TEEPUBLIC_STORE_URL = os.getenv("TEEPUBLIC_STORE_URL", "").strip()

CHANNELS_FILE = "channels.json"
LAST_VIDEO_FILE = "last_video.txt"

YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

if not DISCORD_TOKEN or not GUILD_ID:
    raise SystemExit("The Overseer requires credentials.")

# ================= LOAD CONFIG =================
def load_config():
    if not os.path.exists(CHANNELS_FILE):
        return {}
    with open(CHANNELS_FILE, "r") as f:
        return json.load(f)

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

# ================= CHANNEL SYNC =================
async def sync_channels(guild):
    config = load_config()

    dry_run = config.get("dry_run", False)
    deletion_delay = config.get("deletion_delay_seconds", 0)
    log_channel_name = config.get("log_channel", "overseer-log")
    managed_categories = config.get("managed_categories", {})

    protected_channels = {
        guild.rules_channel,
        guild.public_updates_channel,
        guild.system_channel
    }

    log_channel = get(guild.text_channels, name=log_channel_name)
    if not log_channel:
        log_channel = await guild.create_text_channel(log_channel_name)

    changes = []

    for cat_name, types in managed_categories.items():

        category = get(guild.categories, name=cat_name)
        if not category:
            changes.append(f"Created category: {cat_name}")
            if not dry_run:
                category = await guild.create_category(cat_name)

        desired_text = set(types.get("text", []))
        desired_voice = set(types.get("voice", []))

        # --- CREATE MISSING ---
        if category:
            for name in desired_text:
                if not get(category.text_channels, name=name):
                    changes.append(f"Created text channel: {name}")
                    if not dry_run:
                        await guild.create_text_channel(name, category=category)

            for name in desired_voice:
                if not get(category.voice_channels, name=name):
                    changes.append(f"Created voice channel: {name}")
                    if not dry_run:
                        await guild.create_voice_channel(name, category=category)

            # --- DELETE EXTRA ---
            for channel in category.channels:

                if channel in protected_channels:
                    continue

                if isinstance(channel, discord.TextChannel):
                    if channel.name not in desired_text:
                        changes.append(f"Deleted text channel: {channel.name}")
                        if not dry_run:
                            await asyncio.sleep(deletion_delay)
                            await channel.delete()

                if isinstance(channel, discord.VoiceChannel):
                    if channel.name not in desired_voice:
                        changes.append(f"Deleted voice channel: {channel.name}")
                        if not dry_run:
                            await asyncio.sleep(deletion_delay)
                            await channel.delete()

    # --- POST SUMMARY ---
    if changes:
        summary = "\n".join(changes)
        if dry_run:
            summary = "**DRY RUN MODE — No changes applied**\n\n" + summary
        await log_channel.send(f"Overseer Report:\n\n{summary}")
    else:
        await log_channel.send("Overseer Report:\nNo structural anomalies detected.")

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

    # ---------- YOUTUBE ----------
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
