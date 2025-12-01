import discord
import asyncio
import re
import json
import time
from datetime import datetime, timedelta
from transformers import pipeline

TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID"))

# ------------------------------
#  OPTIMIZED MODELS & FILTERS
# ------------------------------

# Very lightweight and fast toxic classifier
classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    device=-1
)

# Quick regex NSFW media patterns
NSFW_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mov")

BAD_LINK_PATTERNS = [
    r"grabify\.", r"porn", r"sex", r"xxx", r"free nitro", r"discord-nitro",
    r"iplogger", r"click here", r"verify your account"
]

SLUR_WORDS = [
    "fag", "retard", "nigger", "coon", "gook", "slur1", "slur2"
]

SPAM_PATTERNS = [
    r"(.)\1{6,}",       # repeated characters spam
    r"(http.*){5,}",    # link spam
    r"@everyone",       # ping raids
    r"@here"
]

# ------------------------------
#  LOGGING SYSTEM
# ------------------------------

def write_log(entry):
    try:
        with open("moderation_log.json", "r") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(entry)

    with open("moderation_log.json", "w") as f:
        json.dump(logs, f, indent=4)

# ------------------------------
#  MESSAGE SCANNING
# ------------------------------

def is_suspicious_link(msg):
    for p in BAD_LINK_PATTERNS:
        if re.search(p, msg.lower()):
            return True
    return False

def contains_slur(msg):
    m = msg.lower()
    return any(w in m for w in SLUR_WORDS)

def looks_like_spam(msg):
    return any(re.search(p, msg.lower()) for p in SPAM_PATTERNS)

def predict_toxicity(msg):
    res = classifier(msg[:300])[0]  # crop long messages
    return res["label"] == "toxic" and res["score"] >= 0.70

def is_nsfw_filename(name):
    return any(name.lower().endswith(ext) for ext in NSFW_EXTENSIONS)

async def scan_message(message):
    content = message.content.lower()

    flagged = False
    reason = None

    if contains_slur(content):
        flagged = True
        reason = "Slur detected"

    elif looks_like_spam(content):
        flagged = True
        reason = "Spam detected"

    elif is_suspicious_link(content):
        flagged = True
        reason = "Suspicious / dangerous link"

    elif predict_toxicity(content):
        flagged = True
        reason = "Toxic message"

    # Scan attachments
    for attachment in message.attachments:
        if is_nsfw_filename(attachment.filename):
            flagged = True
            reason = "Potential NSFW media"

    if flagged:
        try:
            await message.delete()
        except:
            pass

        write_log({
            "when": str(datetime.utcnow()),
            "author": str(message.author),
            "reason": reason,
            "content": message.content
        })

# ------------------------------
#  NUKE + RAID DETECT
# ------------------------------

NUKE_WINDOW = 60 * 10  # 10 minutes
last_counts = {"channels": 0, "time": time.time()}

async def detect_micro_nuke(guild: discord.Guild):
    global last_counts

    current = len(guild.channels)
    previous = last_counts["channels"]

    if previous > 0:
        diff = abs(current - previous)
        if diff >= 5:
            write_log({
                "when": str(datetime.utcnow()),
                "type": "NUKE",
                "before": previous,
                "after": current,
                "difference": diff
            })

    last_counts["channels"] = current
    last_counts["time"] = time.time()

# ------------------------------
#  MAIN LOOP
# ------------------------------

async def run_advanced_scan():
    intents = discord.Intents.default()
    intents.messages = True
    intents.guilds = True
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"[MOD] Logged in as {client.user}")

        guild = client.get_guild(GUILD_ID)

        if guild is None:
            print("Guild not found!")
            await client.close()
            return

        # Scan last 500 messages per channel
        for channel in guild.text_channels:
            try:
                async for msg in channel.history(limit=500):
                    await scan_message(msg)
            except:
                pass

        # Detect micro nuke
        await detect_micro_nuke(guild)

        print("[MOD] Finished scan")
        await client.close()

    client.run(TOKEN)

# Run instantly when invoked
if __name__ == "__main__":
    asyncio.run(run_advanced_scan())
