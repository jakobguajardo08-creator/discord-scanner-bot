#!/usr/bin/env python3
"""
daily_scan.py

Discord bot for Roblox development group:
- Deletes all channels & categories daily
- Restores channels & categories with dynamic emojis
- Posts AI summarized Reddit news
- Detects nukes
- Scans messages for toxic, NSFW, and suspicious content
"""

import os, json, io, sys, asyncio, random, re
from datetime import datetime
import discord
from discord.utils import get
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForImageClassification, pipeline, AutoModelForSeq2SeqLM, AutoTokenizer as AutoTokenizerSumm
import torch
from PIL import Image

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None
BACKUP_FILE = "guild_backup.json"
DEFAULT_CHANNELS_FILE = "default_channels.json"

MAX_MESSAGES = 200
TOXIC_THRESHOLD = 0.5
NSFW_THRESHOLD = 0.75
SCAN_IMAGE_TYPES = (".png",".jpg",".jpeg",".gif",".webp")
NUKE_DELETION_THRESHOLD = 0.1
NUKE_CREATION_THRESHOLD = 10
SUSPICIOUS_PATTERNS = [
    r"(api_key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"exec\(",
    r"eval\(",
    r"base64\.b64decode",
    r"curl .* --output",
    r"powershell .* -EncodedCommand"
]
SEASONAL_EMOJIS = {"🎃":[10,11], "🎄":[12], "💖":[2], "🌸":[3,4]}

REPORT = {
    "toxic_messages":[],
    "nsfw_attachments":[],
    "suspicious_code":[],
    "nuke_events":[],
    "restoration_actions":[],
    "decorations":[]
}

if not DISCORD_TOKEN or not GUILD_ID:
    print("DISCORD_TOKEN or GUILD_ID not set", file=sys.stderr)
    sys.exit(2)

# ---------------- MODELS ----------------
txt_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
txt_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
TOXIC_LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

img_extractor = AutoFeatureExtractor.from_pretrained("Falconsai/nsfw_image_detection")
img_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")

# Summarization pipeline (BART large)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

# ---------------- CLIENT ----------------
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

# ---------------- UTILITIES ----------------
def load_json(path):
    return json.load(open(path,"r",encoding="utf-8")) if os.path.exists(path) else {}

def save_json(path,data):
    json.dump(data,open(path,"w",encoding="utf-8"),ensure_ascii=False,indent=2)

def get_dynamic_emoji():
    month = datetime.utcnow().month
    for e,months in SEASONAL_EMOJIS.items():
        if month in months: return e
    return "🔹"

# ---------------- REDDIT FETCH ----------------
def fetch_reddit_news(subreddit="worldnews", limit=15):
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t=day"
    headers = {"User-Agent": "DiscordBot-NewsFetcher/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        posts = []
        for p in data["data"]["children"]:
            title = p["data"]["title"]
            posts.append(title)
        return posts
    except:
        return ["Unable to fetch Reddit news today."]

def summarize_news(headlines):
    text = " ".join(headlines)
    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        return summary
    except:
        return "Unable to summarize news today."

# ---------------- TEXT SCAN ----------------
def run_model_on_text(text):
    try:
        inputs = txt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = txt_model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy()
        return {label: float(probs[i]) for i,label in enumerate(TOXIC_LABELS)}
    except: return {label:0.0 for label in TOXIC_LABELS}

def is_text_toxic(text):
    res = run_model_on_text(text)
    for _,score in res.items():
        if score>=TOXIC_THRESHOLD: return True,res
    return False,res

def suspicious_code_check(text):
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat,text,re.IGNORECASE): return pat
    return None

# ---------------- IMAGE SCAN ----------------
def run_model_on_image_bytes(data):
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = img_extractor(images=img, return_tensors="pt")
        outputs = img_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
        return float(probs[1]) if len(probs)>1 else float(probs.max())
    except: return 0.0

# ---------------- DELETE ALL ----------------
async def delete_all_channels_categories(guild):
    for ch in guild.channels:
        try: await ch.delete()
        except: pass
    for cat in guild.categories:
        try: await cat.delete()
        except: pass

# ---------------- RESTORE ----------------
async def restore_channels_roles_dynamic(guild, default_data):
    emoji = get_dynamic_emoji()
    await delete_all_channels_categories(guild)
    category_map = {}
    for cat in default_data.get("categories", []):
        name = cat["name"].replace("{emoji}", emoji)
        try: created_category = await guild.create_category(name)
        except: continue
        category_map[cat["name"]] = created_category
    created_names = set()
    for ch in default_data.get("channels", []):
        channel_name = ch["name"].replace("{emoji}", emoji)
        if channel_name in created_names: continue
        category_obj = category_map.get(ch.get("category"))
        try:
            if ch["type"]=="text":
                await guild.create_text_channel(
                    name=channel_name,
                    topic=ch.get("topic"),
                    nsfw=ch.get("nsfw",False),
                    slowmode_delay=ch.get("slowmode_delay",0),
                    category=category_obj
                )
                created_names.add(channel_name)
        except: continue

# ---------------- MESSAGE SCAN ----------------
async def scan_messages(guild):
    for ch in guild.text_channels:
        perms = ch.permissions_for(guild.me)
        if not perms.read_messages or not perms.read_message_history: continue
        async for msg in ch.history(limit=MAX_MESSAGES,oldest_first=False):
            if msg.author.bot: continue
            if msg.content:
                toxic,scores = is_text_toxic(msg.content)
                if toxic:
                    REPORT["toxic_messages"].append({
                        "channel": ch.name,
                        "author": str(msg.author),
                        "snippet": msg.content[:300],
                        "scores": scores
                    })
                    try: await msg.delete()
                    except: pass
                pat = suspicious_code_check(msg.content)
                if pat:
                    REPORT["suspicious_code"].append({
                        "channel": ch.name,
                        "author": str(msg.author),
                        "pattern": pat,
                        "snippet": msg.content[:300]
                    })
            for att in msg.attachments:
                if any(att.filename.lower().endswith(ext) for ext in SCAN_IMAGE_TYPES):
                    data = None
                    try: data = await att.read()
                    except:
                        try:
                            r = requests.get(att.url, timeout=10)
                            if r.status_code==200: data=r.content
                        except: data=None
                    if data:
                        nsfw_score = run_model_on_image_bytes(data)
                        if nsfw_score>=NSFW_THRESHOLD:
                            REPORT["nsfw_attachments"].append({
                                "channel": ch.name,
                                "author": str(msg.author),
                                "attachment": att.url,
                                "score": nsfw_score
                            })
                            try: await msg.delete()
                            except: pass

# ---------------- DAILY NEWS ----------------
async def post_daily_info(guild):
    channel = get(guild.text_channels, name="daily-info")
    if not channel:
        try:
            channel = await guild.create_text_channel("daily-info", topic="AI-generated daily news summary")
        except: return
    try:
        headlines = fetch_reddit_news()
        summary = summarize_news(headlines)
        await channel.send("📰 **AI Daily News Summary (from Reddit):**")
        await channel.send(summary)
    except: pass

# ---------------- NUKE DETECTION ----------------
async def detect_nuke(guild):
    prev = load_json(BACKUP_FILE)
    old_channels = prev.get("channels",[])
    curr = guild.channels
    deleted = max(0,len(old_channels)-len([ch for ch in curr if ch.name in [c.get("name") for c in old_channels]]))
    created = max(0,len(curr)-len(old_channels))
    if deleted>max(1,int(len(old_channels)*NUKE_DELETION_THRESHOLD)) and created>=NUKE_CREATION_THRESHOLD:
        REPORT["nuke_events"].append({
            "deleted_count": deleted,
            "created_count": created,
            "timestamp": datetime.utcnow().isoformat()
        })
        print("NUKE detected!")

# ---------------- MAIN ----------------
@client.event
async def on_ready():
    try:
        guild = client.get_guild(GUILD_ID)
        if not guild: await client.close(); return
        default_data = load_json(DEFAULT_CHANNELS_FILE)
        await detect_nuke(guild)
        await restore_channels_roles_dynamic(guild,default_data)
        await scan_messages(guild)
        await post_daily_info(guild)
        snapshot = {"roles":[],"channels":[]}
        for role in guild.roles:
            snapshot["roles"].append({
                "name":role.name,
                "permissions":role.permissions.value,
                "hoist":role.hoist,
                "mentionable":role.mentionable
            })
        for ch in guild.channels:
            ch_info={"name":ch.name,"type":str(ch.type),"position":ch.position}
            if isinstance(ch,discord.TextChannel):
                ch_info.update({"topic":ch.topic,"nsfw":ch.nsfw,"slowmode_delay":ch.slowmode_delay})
            snapshot["channels"].append(ch_info)
        save_json(BACKUP_FILE,snapshot)
        print("Daily tasks completed.")
    except Exception as e: print("Error:",e)
    finally: await client.close(); sys.exit(0)

if __name__=="__main__":
    client.run(DISCORD_TOKEN)
