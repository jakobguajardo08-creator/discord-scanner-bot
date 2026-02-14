#!/usr/bin/env python3
import os
import json
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
    raise SystemExit("The Overseer requires DISCORD_TOKEN and GUILD_ID.")

# ================= LOAD CONFIG =================
def load_channel_config():
    if not os.path.exists(CHANNELS_FILE):
        print("The Overseer cannot find channels.json. It will remember this.")
        return {}
    with open(CHANNELS_FILE, "r") as f:
        return json.load(f)

# ================= CHANNEL SYNC =================
async def sync_channels(guild):
    print("Scanning the structure...")
    config = load_channel_config()
    desired_structure = config.get("categories", {})

    desired_channel_names = set()
    desired_categories = set(desired_structure.keys())

    protected_channels = {
        guild.rules_channel,
        guild.public_updates_channel,
        guild.system_channel
    }

    # Create categories + channels
    for cat_name in desired_categories:
        category = get(guild.categories, name=cat_name)
        if not category:
            print(f"Manifesting category: {cat_name}")
            category = await guild.create_category(cat_name)

        for channel_name in desired_structure[cat_name]:
            desired_channel_names.add(channel_name)
            existing = get(guild.text_channels, name=channel_name)

            if not existing:
                print(f"Creating missing channel: {channel_name}")
                await guild.create_text_channel(
                    channel_name,
                    category=category
                )

    # Delete unapproved channels
    for channel in guild.text_channels:

        if channel in protected_channels:
            print(f"Channel {channel.name} is protected. It remains.")
            continue

        if channel.name in desired_channel_names:
            continue

        try:
            print(f"Erasing unauthorized channel: {channel.name}")
            await channel.delete()
        except discord.Forbidden:
            print(f"The Overseer was denied access to {channel.name}. Curious.")
        except Exception as e:
            print(f"Anomaly while deleting {channel.name}: {e}")

    print("Structure stabilized.")

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
            print(f"Ignoring short-form anomaly: {title}")
            continue

        videos.append({
            "video_id": video_id,
            "title": title,
            "url": url
        })

    return videos


def get_last_posted_video():
    if not os.path.exists(LAST_VIDEO_FILE):
        return None
    with open(LAST_VIDEO_FILE, "r") as f:
        return f.read().strip()


def save_last_posted_video(video_id):
    with open(LAST_VIDEO_FILE, "w") as f:
        f.write(video_id)


async def fetch_videos():
    if not YT_CHANNEL_ID:
        return []

    async with aiohttp.ClientSession() as s:
        async with s.get(YOUTUBE_RSS.format(YT_CHANNEL_ID)) as r:
            if r.status != 200:
                print("The feed did not respond.")
                return []
            return parse_rss(await r.read())

# ================= TEEPUBLIC =================
async def fetch_products():
    if not TEEPUBLIC_STORE_URL:
        return []

    headers = {"User-Agent": "Mozilla/5.0"}

    async with aiohttp.ClientSession(headers=headers) as s:
        async with s.get(TEEPUBLIC_STORE_URL) as r:
            if r.status != 200:
                print("The merchandise plane is silent.")
                return []
            html = await r.text()

    soup = BeautifulSoup(html, "html.parser")

    products = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/t-shirt" in href:
            title = link.get("title") or link.text.strip()
            if not title:
                continue
            url = "https://www.teepublic.com" + href
            products.append({"title": title, "url": url})

    seen = set()
    unique = []
    for p in products:
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)

    return unique[:5]

# ================= DISCORD =================
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print("The Overseer has awakened.")
    guild = client.get_guild(GUILD_ID)

    if not guild:
        print("The guild was not found. Something is wrong.")
        await client.close()
        return

    await sync_channels(guild)

    # ---------- YOUTUBE ----------
    yt_channel = get(guild.text_channels, name="externally-hosted-moving-images")
    if yt_channel:
        videos = await fetch_videos()

        if videos:
            newest_video = videos[0]
            last_posted = get_last_posted_video()

            if newest_video["video_id"] != last_posted:
                print(f"Broadcasting: {newest_video['title']}")
                await yt_channel.send(
                    f"{newest_video['title']}\n{newest_video['url']}"
                )
                save_last_posted_video(newest_video["video_id"])
            else:
                print("No new transmissions detected.")

    # ---------- TEEPUBLIC ----------
    tee_channel = get(guild.text_channels, name="commercial-goods")
    if tee_channel:
        products = await fetch_products()
        if products:
            print(f"Publishing merchandise: {products[0]['title']}")
            await tee_channel.send(
                f"Store listing update:\n\n"
                f"{products[0]['title']}\n"
                f"{products[0]['url']}"
            )

    print("The Overseer returns to dormancy.")
    await client.close()

client.run(DISCORD_TOKEN)
