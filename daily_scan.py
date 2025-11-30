#!/usr/bin/env python3
"""
daily_scan.py

Discord daily maintenance bot:
- Deletes and restores channels (using backup + default channels)
- Posts fun facts and today's events
- Adds seasonal emoji to channels and roles
- Detects nukes
- Scans messages for toxicity, NSFW, and suspicious code
"""

import os
import json
import io
import sys
import asyncio
import random
import re
from datetime import datetime, timedelta

import discord
from discord import Permissions
from discord.utils import get

import requests
from bs4 import BeautifulSoup

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None

BACKUP_FILE = "guild_backup.json"
FUN_FACTS_FILE = "daily_facts.json"
DEFAULT_CHANNELS_FILE = "default_channels.json"

MAX_MESSAGES = 200
TOXIC_THRESHOLD = 0.5
NSFW_THRESHOLD = 0.75
SCAN_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".webp")

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

SEASONAL_EMOJIS = {
    "🎃": [10, 11],
    "🎄": [12],
    "💖": [2],
    "🌸": [3, 4],
}

REPORT = {
    "toxic_messages": [],
    "nsfw_attachments": [],
    "suspicious_code": [],
    "nuke_events": [],
    "restoration_actions": [],
    "decorations": []
}

# ---------------- SAFETY ----------------
if not DISCORD_TOKEN or not GUILD_ID:
    print("DISCORD_TOKEN or GUILD_ID not set", file=sys.stderr)
    sys.exit(2)

# ---------------- LOAD HUGGINGFACE MODELS ----------------
print("Loading text toxicity model...")
TEXT_MODEL_NAME = "unitary/toxic-bert"
txt_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
txt_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)
TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

print("Loading NSFW image model...")
IMG_MODEL_NAME = "Falconsai/nsfw_image_detection"
img_extractor = AutoFeatureExtractor.from_pretrained(IMG_MODEL_NAME)
img_model = AutoModelForImageClassification.from_pretrained(IMG_MODEL_NAME)

# ---------------- DISCORD CLIENT ----------------
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

# ---------------- UTILITIES ----------------
def load_json(file_path):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(file_path, data):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def get_seasonal_emoji():
    month = datetime.utcnow().month
    for emoji, months in SEASONAL_EMOJIS.items():
        if month in months:
            return emoji
    return "🔹"

def pick_daily_fact():
    data = load_json(FUN_FACTS_FILE)
    facts = data.get("facts", [])
    return random.choice(facts) if facts else "No fun fact today."

def fetch_today_events():
    try:
        r = requests.get("https://nationaltoday.com/today/", timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            events = [li.get_text(strip=True) for li in soup.select("ul.today-events li")]
            return events[:5] if events else ["No events today."]
    except Exception:
        return ["No events today."]
    return ["No events today."]

def run_model_on_text(text):
    try:
        inputs = txt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = txt_model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy()
        return {label: float(probs[i]) for i, label in enumerate(TOXIC_LABELS)}
    except:
        return {label: 0.0 for label in TOXIC_LABELS}

def is_text_toxic(text):
    res = run_model_on_text(text)
    for label, score in res.items():
        if score >= TOXIC_THRESHOLD:
            return True, res
    return False, res

def suspicious_code_check(text):
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    return None

def run_model_on_image_bytes(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = img_extractor(images=img, return_tensors="pt")
        outputs = img_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
        nsfw_score = float(probs[1]) if len(probs) > 1 else float(probs.max())
        return nsfw_score
    except:
        return 0.0

# ---------------- RESTORE / RESET ----------------
async def restore_channels_roles(guild, backup, default_channels):
    emoji = get_seasonal_emoji()

    # Restore roles
    existing_roles = {r.name: r for r in guild.roles}
    for role in backup.get("roles", []):
        if role["name"] == "@everyone": continue
        role_name = f"{emoji}-{role['name']}" if not role["name"].startswith(emoji) else role["name"]
        if role_name not in existing_roles:
            try:
                perms = Permissions(role.get("permissions", 0))
                await guild.create_role(name=role_name, permissions=perms,
                                        hoist=role.get("hoist", False),
                                        mentionable=role.get("mentionable", False),
                                        reason="Restore role")
            except: continue

    # Delete all channels
    for ch in guild.channels:
        try: await ch.delete(reason="Daily reset")
        except: continue

    # Restore categories first
    categories = {}
    for ch in backup.get("channels", []) + default_channels.get("channels", []):
        if ch.get("type") == "category":
            try:
                cat = await guild.create_category(f"{emoji}-{ch['name']}")
                categories[ch.get("id")] = cat
            except: continue

    # Restore text channels
    created_names = set()
    for ch in backup.get("channels", []) + default_channels.get("channels", []):
        if ch.get("type") != "text": continue
        name = f"{emoji}-{ch['name']}" if not ch["name"].startswith(emoji) else ch["name"]
        if name in created_names: continue
        category = categories.get(ch.get("category_id"))
        try:
            await guild.create_text_channel(name=name, topic=ch.get("topic"),
                                            nsfw=ch.get("nsfw", False),
                                            slowmode_delay=ch.get("slowmode_delay", 0),
                                            category=category,
                                            reason="Daily reset + restoration")
            created_names.add(name)
        except: continue

# ---------------- MESSAGE SCAN ----------------
async def scan_messages(guild):
    for ch in guild.text_channels:
        if not ch.permissions_for(guild.me).read_messages: continue
        if not ch.permissions_for(guild.me).read_message_history: continue
        async for msg in ch.history(limit=MAX_MESSAGES, oldest_first=False):
            if msg.author.bot: continue
            if msg.content:
                toxic, scores = is_text_toxic(msg.content)
                if toxic:
                    REPORT["toxic_messages"].append({"channel": ch.name, "author": str(msg.author),
                                                     "snippet": msg.content[:300], "scores": scores})
                    try: await msg.delete()
                    except: pass
                pat = suspicious_code_check(msg.content)
                if pat:
                    REPORT["suspicious_code"].append({"channel": ch.name, "author": str(msg.author),
                                                      "pattern": pat, "snippet": msg.content[:300]})
            for att in msg.attachments:
                if any(att.filename.lower().endswith(ext) for ext in SCAN_IMAGE_TYPES):
                    data = None
                    try: data = await att.read()
                    except:
                        try:
                            r = requests.get(att.url, timeout=10)
                            if r.status_code == 200: data = r.content
                        except: data = None
                    if data:
                        nsfw_score = run_model_on_image_bytes(data)
                        if nsfw_score >= NSFW_THRESHOLD:
                            REPORT["nsfw_attachments"].append({"channel": ch.name, "author": str(msg.author),
                                                               "attachment": att.url, "score": nsfw_score})
                            try: await msg.delete()
                            except: pass

# ---------------- DAILY INFO ----------------
async def post_daily_info(guild):
    daily_fact = pick_daily_fact()
    events = fetch_today_events()
    info_channel = get(guild.text_channels, name="daily-info")
    if not info_channel:
        try:
            info_channel = await guild.create_text_channel("daily-info", topic="Daily facts & news",
                                                           reason="Daily info channel")
        except: return
    try:
        await info_channel.send(f"📅 **Daily Fun Fact:** {daily_fact}")
        await info_channel.send("🌎 **Today's Events / Holidays:**\n" + "\n".join([f"- {e}" for e in events]))
    except: pass

# ---------------- NUKE DETECTION ----------------
async def detect_nuke(guild):
    prev_snapshot = load_json(BACKUP_FILE)
    old_channels = prev_snapshot.get("channels", [])
    current_channels = guild.channels
    deleted_count = max(0, len(old_channels) - len([ch for ch in current_channels if ch.id in [c.get("id") for c in old_channels]]))
    created_count = max(0, len(current_channels) - len(old_channels))
    if deleted_count > max(1, int(len(old_channels) * NUKE_DELETION_THRESHOLD)) and created_count >= NUKE_CREATION_THRESHOLD:
        REPORT["nuke_events"].append({"deleted_count": deleted_count, "created_count": created_count,
                                      "timestamp": datetime.utcnow().isoformat()})
        print("NUKE detected!")

# ---------------- MAIN DAILY TASK ----------------
@client.event
async def on_ready():
    try:
        guild = client.get_guild(GUILD_ID)
        if not guild:
            print(f"Bot not in guild {GUILD_ID}")
            await client.close(); return
        print(f"Connected as {client.user} to guild {guild.name}")

        # Load backups
        backup = load_json(BACKUP_FILE)
        default_channels = load_json(DEFAULT_CHANNELS_FILE)

        # Detect nukes
        await detect_nuke(guild)

        # Reset & restore
        await restore_channels_roles(guild, backup, default_channels)

        # Scan messages
        await scan_messages(guild)

        # Post daily info
        await post_daily_info(guild)

        # Update backup
        # Capture current guild state for future restores
        snapshot = {"roles": [], "channels": []}
        for role in guild.roles:
            snapshot["roles"].append({"name": role.name, "permissions": role.permissions.value,
                                      "hoist": role.hoist, "mentionable": role.mentionable})
        for ch in guild.channels:
            ch_info = {"name": ch.name, "type": str(ch.type), "position": ch.position}
            if isinstance(ch, discord.TextChannel):
                ch_info.update({"topic": ch.topic, "nsfw": ch.nsfw, "slowmode_delay": ch.slowmode_delay})
            snapshot["channels"].append(ch_info)
        save_json(BACKUP_FILE, snapshot)

        print("Daily tasks completed.")

    except Exception as e:
        print("Error:", e)
        import traceback; traceback.print_exc()
    finally:
        await client.close()
        sys.exit(0)

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
