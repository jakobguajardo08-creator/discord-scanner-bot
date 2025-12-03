#!/usr/bin/env python3
"""
daily_scan.py — Lightweight Discord Daily Maintenance Bot

Features:
- Deletes ONLY channels/categories not in default configuration (anti-nuke recovery)
- Deletes toxic, NSFW, controversial, extremist, or harmful messages
- Summarizes unbiased daily news from multiple sources
- Fast execution, optimized for GitHub CI runners
"""

import os, json, io, sys, asyncio, re
from datetime import datetime
import discord
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForImageClassification, pipeline

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None

DEFAULT_CHANNELS_FILE = "default_channels.json"
SCAN_LIMIT = 150

TOXIC_THRESHOLD = 0.48
NSFW_THRESHOLD = 0.70

SCAN_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")

# Controversial / extremist / political spikes
CONTROVERSIAL_PATTERNS = [
    r"\b(hitler|nazi|heil)\b",
    r"\bKKK\b",
    r"\bright-wing|left-wing|extremist\b",
    r"\bgenocide\b",
    r"\bdoxxing\b",
    r"\bterror(ist|ism)\b",
]

# ---------------- MODELS (lightweight versions) ----------------
txt_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
txt_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

img_extractor = AutoFeatureExtractor.from_pretrained("Falconsai/nsfw_image_detection")
img_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

# Lightweight summarizer
summarizer = pipeline("summarization",
                      model="sshleifer/distilbart-cnn-12-6",
                      framework="pt")

intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

# ---------------- UTIL ----------------
def load_json(path):
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}

def is_toxic(text):
    inputs = txt_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    outputs = txt_model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy()
    return max(probs) >= TOXIC_THRESHOLD

def is_controversial(text):
    for p in CONTROVERSIAL_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def is_nsfw_image(data):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    inputs = img_extractor(images=img, return_tensors="pt")
    outputs = img_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
    return float(probs[1]) >= NSFW_THRESHOLD

# ---------------- NEWS FETCH ----------------
def reddit_top():
    try:
        r = requests.get("https://www.reddit.com/r/worldnews/top.json?limit=10&t=day",
                         headers={"User-Agent": "NewsBot"}, timeout=8)
        return [p["data"]["title"] for p in r.json()["data"]["children"]]
    except:
        return []

def hackernews_top():
    try:
        r = requests.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=8)
        ids = r.json()[:10]
        titles = []
        for i in ids:
            item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json", timeout=5).json()
            if item and "title" in item: titles.append(item["title"])
        return titles
    except:
        return []

def ap_news():
    try:
        r = requests.get("https://feeds.apnews.com/apf-topnews", timeout=8)
        return re.findall(r"<title>(.*?)</title>", r.text)[1:11]
    except:
        return []

def merge_unbiased(stories):
    unique = []
    for s in stories:
        if not any(s.lower() in o.lower() or o.lower() in s.lower() for o in unique):
            unique.append(s)
    return unique

def summarize_stories(stories):
    text = " ".join(stories)
    try:
        return summarizer(text, max_length=140, min_length=50, do_sample=False)[0]["summary_text"]
    except:
        return "Unable to summarize today's world news."

# ---------------- RESTORE ----------------
async def restore_structure(guild, default):
    allowed_cats = {c["name"] for c in default.get("categories", [])}
    allowed_channels = {c["name"] for c in default.get("channels", [])}

    # Delete anything that does NOT belong
    for ch in guild.channels:
        if ch.name not in allowed_channels and ch.category and ch.category.name not in allowed_cats:
            try: await ch.delete()
            except: pass

    for cat in guild.categories:
        if cat.name not in allowed_cats:
            try: await cat.delete()
            except: pass

    # Recreate if needed
    name_to_cat = {}
    for c in default.get("categories", []):
        existing = discord.utils.get(guild.categories, name=c["name"])
        if existing: 
            name_to_cat[c["name"]] = existing
        else:
            name_to_cat[c["name"]] = await guild.create_category(c["name"])

    for c in default.get("channels", []):
        if not discord.utils.get(guild.channels, name=c["name"]):
            parent = name_to_cat.get(c.get("category"))
            await guild.create_text_channel(c["name"], category=parent)

# ---------------- SCAN ----------------
async def scan_messages(guild):
    for ch in guild.text_channels:
        perms = ch.permissions_for(guild.me)
        if not perms.read_messages: continue

        async for msg in ch.history(limit=SCAN_LIMIT):
            if msg.author.bot: continue

            # TEXT CHECK
            if msg.content:
                if is_toxic(msg.content) or is_controversial(msg.content):
                    try: await msg.delete()
                    except: pass
                    continue

            # IMAGE CHECK
            for att in msg.attachments:
                if any(att.filename.lower().endswith(ext) for ext in SCAN_IMAGE_TYPES):
                    data = await att.read()
                    if is_nsfw_image(data):
                        try: await msg.delete()
                        except: pass

# ---------------- DAILY INFO ----------------
async def post_daily(guild):
    ch = discord.utils.get(guild.text_channels, name="daily-info")
    if not ch:
        ch = await guild.create_text_channel("daily-info")

    combined = merge_unbiased(reddit_top() + hackernews_top() + ap_news())
    summary = summarize_stories(combined)

    await ch.send("📰 **Daily Unbiased Summary**")
    await ch.send(summary)

# ---------------- MAIN ----------------
@client.event
async def on_ready():
    guild = client.get_guild(GUILD_ID)
    if not guild:
        await client.close()
        return

    default = load_json(DEFAULT_CHANNELS_FILE)

    await restore_structure(guild, default)
    await scan_messages(guild)
    await post_daily(guild)

    print("Daily tasks complete.")
    await client.close()
    sys.exit(0)

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
