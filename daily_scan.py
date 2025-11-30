#!/usr/bin/env python3
"""
daily_scan.py

Daily Discord server scanner & recovery tool.
- Scans for toxic messages, NSFW attachments, suspicious code.
- Detects server nukes and restores channels/roles from backup.
- Adds seasonal decorations (emoji) to channels/roles.
"""

import os
import json
import re
import io
import sys
import time
import asyncio
import traceback
from datetime import datetime

import discord
from discord import Permissions

# HuggingFace models
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForImageClassification
import torch
from PIL import Image
import requests

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None
BACKUP_FILENAME = f"backup_{GUILD_ID}.json" if GUILD_ID else "backup_unknown.json"

MAX_MESSAGES_PER_CHANNEL = 200
TOXIC_THRESHOLD = 0.5
NSFW_THRESHOLD = 0.75
SCAN_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".webp")

SEASONAL_EMOJIS = {
    "🎃": [10, 11],   # Oct-Nov: Halloween
    "🎄": [12],       # Dec: Christmas
    "💖": [2],        # Feb: Valentine's
    "🌸": [3, 4],     # Spring
}

SUSPICIOUS_PATTERNS = [
    r"(api_key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"exec\(",
    r"eval\(",
    r"base64\.b64decode",
    r"curl .* --output",
    r"powershell .* -EncodedCommand"
]

REPORT = {
    "toxic_messages": [],
    "nsfw_attachments": [],
    "suspicious_code": [],
    "nuke_events": [],
    "restoration_actions": [],
    "decorations": []
}

# ---------------- SAFETY CHECKS ----------------
if not DISCORD_TOKEN:
    print("ERROR: DISCORD_TOKEN not set.", file=sys.stderr)
    sys.exit(2)
if not GUILD_ID:
    print("ERROR: GUILD_ID not set.", file=sys.stderr)
    sys.exit(2)

# ---------------- LOAD MODELS ----------------
print("Loading text toxicity model...")
TEXT_MODEL_NAME = "unitary/toxic-bert"
txt_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
txt_model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)

print("Loading image NSFW model...")
IMG_MODEL_NAME = "Falconsai/nsfw_image_detection"
img_extractor = AutoFeatureExtractor.from_pretrained(IMG_MODEL_NAME)
img_model = AutoModelForImageClassification.from_pretrained(IMG_MODEL_NAME)

TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# ---------------- DISCORD CLIENT ----------------
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True

client = discord.Client(intents=intents)

# ---------------- UTILITIES ----------------
def run_model_on_text(text):
    try:
        inputs = txt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = txt_model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy()
        return {label: float(probs[i]) for i, label in enumerate(TOXIC_LABELS)}
    except Exception as e:
        print("Text model error:", e)
        return {label: 0.0 for label in TOXIC_LABELS}

def is_text_toxic(text, threshold=TOXIC_THRESHOLD):
    res = run_model_on_text(text)
    for label, score in res.items():
        if score >= threshold:
            return True, res
    return False, res

def run_model_on_image_bytes(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = img_extractor(images=img, return_tensors="pt")
        outputs = img_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
        nsfw_score = float(probs[1]) if len(probs) > 1 else float(probs.max())
        return nsfw_score
    except Exception as e:
        print("Image model error:", e)
        return 0.0

def suspicious_code_check(text):
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    return None

def load_backup():
    if os.path.exists(BACKUP_FILENAME):
        try:
            with open(BACKUP_FILENAME, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("Failed to read backup file:", e)
            return {}
    return {}

def save_backup_locally(data):
    try:
        with open(BACKUP_FILENAME, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("Backup saved locally:", BACKUP_FILENAME)
        return True
    except Exception as e:
        print("Failed to write backup locally:", e)
        return False

# ---------------- SEASONAL DECORATION ----------------
def get_seasonal_emoji():
    month = datetime.utcnow().month
    for emoji, months in SEASONAL_EMOJIS.items():
        if month in months:
            return emoji
    return None

async def decorate_channels_and_roles(guild):
    emoji = get_seasonal_emoji()
    if not emoji:
        return
    try:
        # Add emoji to text channels
        for ch in guild.text_channels:
            if not ch.name.startswith(emoji):
                new_name = f"{emoji}-{ch.name}"
                try:
                    await ch.edit(name=new_name)
                    REPORT["decorations"].append({"channel": ch.name, "new_name": new_name})
                except Exception:
                    continue
        # Add emoji to roles
        for role in guild.roles:
            if role.name not in ["@everyone"] and not role.name.startswith(emoji):
                new_name = f"{emoji}-{role.name}"
                try:
                    await role.edit(name=new_name)
                    REPORT["decorations"].append({"role": role.name, "new_name": new_name})
                except Exception:
                    continue
    except Exception as e:
        print("Decoration error:", e)

# ---------------- SCANNER ----------------
async def scan_guild(guild):
    print("Scanning guild:", guild.name, guild.id)

    # Snapshot roles and channels
    snapshot = {"guild_id": guild.id, "fetched_at": datetime.utcnow().isoformat(), "roles": [], "channels": []}
    for role in sorted(guild.roles, key=lambda r: r.position):
        snapshot["roles"].append({
            "id": role.id, "name": role.name, "color": role.color.value if role.color else 0,
            "permissions": role.permissions.value, "hoist": role.hoist,
            "mentionable": role.mentionable, "position": role.position
        })
    for ch in guild.channels:
        ch_info = {"id": ch.id, "name": ch.name, "type": str(ch.type), "position": ch.position}
        if isinstance(ch, discord.TextChannel):
            ch_info.update({"topic": ch.topic, "nsfw": ch.nsfw, "slowmode_delay": ch.slowmode_delay,
                            "category_id": ch.category.id if ch.category else None})
        snapshot["channels"].append(ch_info)

    previous = load_backup()
    if not previous:
        save_backup_locally(snapshot)
        return REPORT

    # Scan messages
    for ch in guild.text_channels:
        if not ch.permissions_for(guild.me).read_messages or not ch.permissions_for(guild.me).read_message_history:
            continue
        async for msg in ch.history(limit=MAX_MESSAGES_PER_CHANNEL, oldest_first=False):
            if msg.author.bot:
                continue

            # Text toxicity
            if msg.content:
                toxic, scores = is_text_toxic(msg.content)
                if toxic:
                    REPORT["toxic_messages"].append({
                        "channel": ch.name, "author": str(msg.author),
                        "content_snippet": msg.content[:300], "scores": scores
                    })
                    try: await msg.delete()
                    except Exception: pass

                # Suspicious code
                pat = suspicious_code_check(msg.content)
                if pat:
                    REPORT["suspicious_code"].append({
                        "channel": ch.name, "author": str(msg.author),
                        "pattern": pat, "content_snippet": msg.content[:300]
                    })

            # Attachments
            for att in msg.attachments:
                if any(att.filename.lower().endswith(ext) for ext in SCAN_IMAGE_TYPES):
                    data = None
                    try:
                        data = await att.read()
                    except Exception:
                        try:
                            r = requests.get(att.url, timeout=15)
                            if r.status_code == 200:
                                data = r.content
                        except Exception:
                            data = None
                    if data:
                        try:
                            nsfw_score = run_model_on_image_bytes(data)
                            if nsfw_score >= NSFW_THRESHOLD:
                                REPORT["nsfw_attachments"].append({
                                    "channel": ch.name, "author": str(msg.author),
                                    "attachment": att.url, "score": nsfw_score
                                })
                                try: await msg.delete()
                                except Exception: pass
                        except Exception as e:
                            print("Image classification error:", e)

    # Detect nukes
    prev_roles = {r["id"]: r for r in previous.get("roles", [])}
    prev_channels = {c["id"]: c for c in previous.get("channels", [])}
    now_roles = {r["id"]: r for r in snapshot["roles"]}
    now_channels = {c["id"]: c for c in snapshot["channels"]}
    deleted_roles = [pid for pid in prev_roles if pid not in now_roles]
    deleted_channels = [pid for pid in prev_channels if pid not in now_channels]
    if len(deleted_roles) >= max(1, len(prev_roles)//10) or len(deleted_channels) >= max(1, len(prev_channels)//10):
        REPORT["nuke_events"].append({"deleted_roles": deleted_roles, "deleted_channels": deleted_channels, "detected_at": datetime.utcnow().isoformat()})
        await restore_from_snapshot(guild, previous)

    # Update backup if no nuke
    if not REPORT["nuke_events"]:
        save_backup_locally(snapshot)

    # Seasonal decoration
    await decorate_channels_and_roles(guild)

    return REPORT

# ---------------- RESTORE ----------------
async def restore_from_snapshot(guild, snapshot):
    print("Restoring guild from snapshot...")
    restored = {"roles": [], "channels": []}
    existing_role_names = {r.name: r for r in guild.roles}
    for role_info in snapshot.get("roles", []):
        if role_info["name"] == "@everyone": continue
        if role_info["name"] not in existing_role_names:
            try:
                perms = Permissions(role_info.get("permissions", 0))
                new_role = await guild.create_role(name=role_info["name"][:100], permissions=perms,
                                                   hoist=role_info.get("hoist", False),
                                                   mentionable=role_info.get("mentionable", False),
                                                   reason="Restore after nuke")
                restored["roles"].append({"name": new_role.name, "id": new_role.id})
            except Exception:
                continue
    REPORT["restoration_actions"].append({"timestamp": datetime.utcnow().isoformat(), "restored": restored})
    print("Restoration done.")

# ---------------- ENTRY POINT ----------------
@client.event
async def on_ready():
    try:
        guild = client.get_guild(GUILD_ID)
        if not guild:
            print(f"Bot not in guild {GUILD_ID}")
            await client.close()
            return
        print(f"Connected as {client.user}, scanning {guild.name}")
        await scan_guild(guild)

        outname = f"scan_report_{GUILD_ID}_{int(time.time())}.json"
        with open(outname, "w", encoding="utf-8") as of:
            json.dump(REPORT, of, ensure_ascii=False, indent=2)
        print("Report written to", outname)

    except Exception as e:
        print("Main error:", e)
        traceback.print_exc()
    finally:
        await client.close()
        sys.exit(0)

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
