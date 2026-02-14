#!/usr/bin/env python3
import os
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

YT_TARGET_CHANNEL = "externally-hosted-moving-images"
TEEPUBLIC_CHANNEL = "commercial-goods"

YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"
LAST_VIDEO_FILE = "last_video.txt"

if not DISCORD_TOKEN or not GUILD_ID:
    raise SystemExit("Missing DISCORD_TOKEN or GUILD_ID")

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

        # 🚫 Skip Shorts
        if "/shorts/" in url.lower():
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
                return []
            return parse_rss(await r.read())


# ================= TEEPUBLIC =================
async def fetch_products():
    if not TEEPUBLIC_STORE_URL:
        return []

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    async with aiohttp.ClientSession(headers=headers) as s:
        async with s.get(TEEPUBLIC_STORE_URL) as r:
            if r.status != 200:
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

    # Remove duplicates
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
    guild = client.get_guild(GUILD_ID)
    if not guild:
        await client.close()
        return

    # ---------- YOUTUBE ----------
    yt_channel = get(guild.text_channels, name=YT_TARGET_CHANNEL)
    if yt_channel:
        videos = await fetch_videos()

        if videos:
            newest_video = videos[0]  # newest non-short
            last_posted = get_last_posted_video()

            if newest_video["video_id"] != last_posted:
                await yt_channel.send(
                    f"{newest_video['title']}\n{newest_video['url']}"
                )
                save_last_posted_video(newest_video["video_id"])

    # ---------- TEEPUBLIC ----------
    tee_channel = get(guild.text_channels, name=TEEPUBLIC_CHANNEL)
    if tee_channel:
        products = await fetch_products()
        if products:
            await tee_channel.send(
                f"Store listing update:\n\n"
                f"{products[0]['title']}\n"
                f"{products[0]['url']}"
            )

    await client.close()

client.run(DISCORD_TOKEN)
