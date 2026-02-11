#!/usr/bin/env python3
import os
import json
import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from datetime import date
from typing import Optional, Dict, List
from bs4 import BeautifulSoup
import discord
from discord.utils import get

# ================= CONFIG =================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GUILD_ID = int(os.getenv("GUILD_ID", "0"))

YT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID", "").strip()
TEEPUBLIC_STORE_URL = os.getenv("TEEPUBLIC_STORE_URL", "").strip()

STATE_FILE = "lucha_state.json"

YT_TARGET_CHANNEL = "externally-hosted-moving-images"
TEEPUBLIC_CHANNEL = "commercial-goods"

YOUTUBE_RSS = "https://www.youtube.com/feeds/videos.xml?channel_id={}"

if not DISCORD_TOKEN or not GUILD_ID:
    raise SystemExit("Missing DISCORD_TOKEN or GUILD_ID")

# ================= STATE =================
def today_key():
    return f"daily:{date.today().isoformat()}"

def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2)

# ================= YOUTUBE =================
def parse_rss(xml_bytes: bytes) -> List[Dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "yt": "http://www.youtube.com/xml/schemas/2015"
    }

    videos = []
    for entry in root.findall("a:entry", ns):
        videos.append({
            "video_id": entry.find("yt:videoId", ns).text,
            "title": entry.find("a:title", ns).text,
            "url": entry.find("a:link", ns).attrib["href"]
        })
    return videos

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

    async with aiohttp.ClientSession() as s:
        async with s.get(TEEPUBLIC_STORE_URL) as r:
            if r.status != 200:
                return []
            html = await r.text()

    soup = BeautifulSoup(html, "html.parser")
    products = []

    for card in soup.select("a[href*='/t-shirt']")[:10]:
        title = card.get("title") or card.text.strip()
        url = "https://www.teepublic.com" + card.get("href")
        products.append({"title": title, "url": url})

    return products

# ================= DISCORD =================
intents = discord.Intents.default()
intents.guilds = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    state = load_state()
    if state.get(today_key()):
        await client.close()
        return

    guild = client.get_guild(GUILD_ID)
    if not guild:
        await client.close()
        return

    # -------- YOUTUBE --------
    yt_channel = get(guild.text_channels, name=YT_TARGET_CHANNEL)
    videos = await fetch_videos()
    posted_ids = set(state.get("posted_videos", []))

    if yt_channel:
        for video in videos:
            if video["video_id"] not in posted_ids:
                await yt_channel.send(
                    f"{video['title']}\n{video['url']}"
                )
                posted_ids.add(video["video_id"])
                break

    state["posted_videos"] = list(posted_ids)

    # -------- TEEPUBLIC ROTATION --------
    tee_channel = get(guild.text_channels, name=TEEPUBLIC_CHANNEL)
    products = await fetch_products()

    if tee_channel and products:
        index = state.get("merch_index", 0) % len(products)
        product = products[index]

        await tee_channel.send(
            f"Daily merchandise spotlight:\n\n"
            f"{product['title']}\n"
            f"{product['url']}"
        )

        state["merch_index"] = index + 1

    # -------- FINALIZE --------
    state[today_key()] = True
    save_state(state)
    await client.close()

client.run(DISCORD_TOKEN)
