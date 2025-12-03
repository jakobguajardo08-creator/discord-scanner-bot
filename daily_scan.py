#!/usr/bin/env python3
"""
daily_scan.py — Lightweight Discord Daily Maintenance Bot (single file)

Design goals:
- Light & fast: avoid heavy local models; use remote AI API for generation
- Conservative moderation: only delete clearly-violating content or clearly-foreign channels after a detected nuke/raid
- Fun daily features: AI-driven chat summary (funny), daily stats, meme line of the day, lucky number, server mood
- GitHub-ready: easy to run in Actions or a small VM

CONFIG (via environment):
- DISCORD_TOKEN (required)
- GUILD_ID (required)
- DEFAULT_CHANNELS_FILE (optional, default: default_channels.json)
- AI_API_URL (required for AI features) - POST JSON {"prompt": "...", "max_tokens": N}
- AI_API_KEY (optional, used in Authorization header)
- MODERATION_API_URL (optional) - if present, used to check content
- REPORT_FILE (optional, default: mod_report.json)
"""

import os
import sys
import json
import io
import re
import random
import asyncio
from datetime import datetime, timezone
from collections import Counter, defaultdict

import aiohttp
import discord
from discord.utils import get

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None
DEFAULT_CHANNELS_FILE = os.getenv("DEFAULT_CHANNELS_FILE", "default_channels.json")
REPORT_FILE = os.getenv("REPORT_FILE", "mod_report.json")

AI_API_URL = os.getenv("AI_API_URL") or os.getenv("MODEL_API_URL")
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("MODEL_API_KEY")
MODERATION_API_URL = os.getenv("MODERATION_API_URL", None)  # optional

SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "150"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "250"))
NUKE_DELETION_THRESHOLD = float(os.getenv("NUKE_DELETION_THRESHOLD", "0.12"))  # fraction of channels deleted to call nuke
NUKE_NEW_CHANNEL_THRESHOLD = int(os.getenv("NUKE_NEW_CHANNEL_THRESHOLD", "6"))

# thresholds for heuristic checks (no heavy model)
TOXIC_KEYWORDS = ["idiot","kill","die","stfu","retard","faggot","nigger","terrorist"]  # example; moderate yourself
CONTROVERSIAL_PATTERNS = [
    r"\b(hitler|nazi|heil)\b", r"\bKKK\b", r"\bgenocide\b", r"\bdoxx(le|ing)?\b", r"\bterror(ist|ism)\b"
]
SUSPICIOUS_CODE_PATTERNS = [
    r"(api_key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"exec\(", r"eval\(", r"base64\.b64decode", r"curl .* --output",
    r"powershell .* -EncodedCommand", r"wget .* -O"
]

SCAN_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

# Fun phrases (used as fallback if AI unavailable)
FUN_MEMES = [
    "Your daily reminder: Touching grass is free.",
    "If today was a sandwich, it'd be 40% bread.",
    "Server mood: suspiciously caffeinated.",
    "If you read this, you owe the bot 3 cookies."
]

FORTUNES = [
    "The universe is mildly on your side today.",
    "Expect something oddly satisfying later.",
    "A small inconvenience today leads to a funny memory tomorrow."
]

REPORT = {
    "toxic_messages": [],
    "deleted_channels": [],
    "nuke_events": [],
    "daily_summary": {},
    "errors": []
}

# ---------------- DISCORD CLIENT ----------------
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True
intents.members = True
client = discord.Client(intents=intents)

# ---------------- UTIL ----------------
def load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        REPORT["errors"].append(f"load_json_error:{path}:{str(e)}")
    return {}

def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # last-ditch console
        print("Failed to save json:", path, e, file=sys.stderr)

def now_iso():
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

# ---------------- AI CLIENT (light wrapper) ----------------
async def ai_call(prompt: str, max_tokens: int = 300) -> str:
    """
    Calls configured AI API. Expects POST JSON {"prompt": "...", "max_tokens": N}
    Response must contain a top-level string field 'text' or 'result' or 'output'.
    If AI_API_URL is not configured, returns a fallback string.
    """
    if not AI_API_URL:
        # fallback: return a short deterministic fallback
        return "AI not configured — " + (random.choice(FUN_MEMES))
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if AI_API_KEY:
        headers["Authorization"] = f"Bearer {AI_API_KEY}"
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(AI_API_URL, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    REPORT["errors"].append(f"ai_call_http_{resp.status}")
                    text = await resp.text()
                    return f"AI call failed (status {resp.status}): {text[:200]}"
                data = await resp.json()
                # common return fields
                for key in ("text","result","output","response"):
                    if key in data and isinstance(data[key], str):
                        return data[key].strip()
                # sometimes it's in choices[0].text
                if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                    c = data["choices"][0]
                    if isinstance(c, dict) and "text" in c:
                        return c["text"].strip()
                # fallback to stringified object if nothing matched
                return str(data)
    except Exception as e:
        REPORT["errors"].append(f"ai_call_exception:{str(e)}")
        return "AI unavailable (error)."

# ---------------- MODERATION HELPERS (lightweight) ----------------
def heuristic_text_flags(text: str) -> dict:
    """
    Very lightweight heuristics for identifying problematic text.
    Returns dict with flags and short reasons.
    """
    flags = {"toxic": False, "controversial": False, "suspicious_code": None}
    lower = text.lower()
    # toxic keywords
    for bad in TOXIC_KEYWORDS:
        if bad in lower:
            flags["toxic"] = True
            break
    # controversial regex
    for patt in CONTROVERSIAL_PATTERNS:
        if re.search(patt, text, re.IGNORECASE):
            flags["controversial"] = True
            break
    # suspicious code
    for patt in SUSPICIOUS_CODE_PATTERNS:
        if re.search(patt, text, re.IGNORECASE):
            flags["suspicious_code"] = patt
            break
    return flags

async def external_moderation_check(text: str) -> dict:
    """
    Optional moderation API: POST {"text": "..."} -> returns JSON with keys like 'toxic', 'nsfw'
    If MODERATION_API_URL not provided, return empty dict.
    """
    if not MODERATION_API_URL:
        return {}
    headers = {"Content-Type": "application/json"}
    try:
        timeout = aiohttp.ClientTimeout(total=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(MODERATION_API_URL, json={"text": text}, headers=headers) as resp:
                if resp.status != 200:
                    REPORT["errors"].append(f"mod_api_http_{resp.status}")
                    return {}
                return await resp.json()
    except Exception as e:
        REPORT["errors"].append(f"mod_api_exception:{str(e)}")
        return {}

def filename_suspicious(fname: str) -> bool:
    lower = fname.lower()
    if "nsfw" in lower or "porn" in lower or "sexy" in lower:
        return True
    return False

# ---------------- ANTI-NUKE / SNAPSHOT ----------------
def build_snapshot(guild: discord.Guild) -> dict:
    snapshot = {"ts": now_iso(), "channels": [], "categories": []}
    for cat in guild.categories:
        snapshot["categories"].append({"id": cat.id, "name": cat.name})
    for ch in guild.channels:
        snapshot["channels"].append({"id": ch.id, "name": ch.name, "type": str(ch.type)})
    return snapshot

def detect_nuke(prev_snapshot: dict, guild: discord.Guild) -> dict:
    prev_names = [c.get("name") for c in prev_snapshot.get("channels", [])] if prev_snapshot else []
    curr_names = [c.name for c in guild.channels]
    missing = [n for n in prev_names if n not in curr_names]
    new = [c.name for c in guild.channels if c.name not in prev_names]
    result = {
        "deleted_count": len(missing),
        "created_count": len(new),
        "missing_sample": missing[:10],
        "new_sample": new[:10],
        "is_nuke": False
    }
    if prev_names and len(missing) > max(1, int(len(prev_names) * NUKE_DELETION_THRESHOLD)) and len(new) >= NUKE_NEW_CHANNEL_THRESHOLD:
        result["is_nuke"] = True
    return result

async def remediate_foreign_channels(guild: discord.Guild, prev_snapshot: dict):
    """
    Conservative deletion: delete channels that are not in snapshot AND match suspicious heuristics
    Returns list of deleted channel names.
    """
    prev_names = {c.get("name") for c in prev_snapshot.get("channels", [])} if prev_snapshot else set()
    deleted = []
    try:
        for ch in list(guild.channels):
            if ch.name in prev_names:
                continue
            # suspicious patterns in name
            if re.search(r"(spam|raid|free-|giveaway|hacked|nuke|bot-)", ch.name, flags=re.IGNORECASE) or (len(guild.channels) > (len(prev_names) + 8)):
                try:
                    await ch.delete(reason="Detected foreign channel after suspected nuke/raid")
                    deleted.append(ch.name)
                except Exception as e:
                    REPORT["errors"].append(f"failed_delete_channel_{ch.name}:{str(e)}")
    except Exception as e:
        REPORT["errors"].append(f"remediate_exception:{str(e)}")
    return deleted

# ---------------- MESSAGE SCAN & CLEANUP ----------------
async def scan_and_clean_messages(guild: discord.Guild):
    """
    Scan recent messages per channel, delete clearly violating messages.
    Uses lightweight heuristics + optional external moderation API.
    """
    for ch in guild.text_channels:
        perms = ch.permissions_for(guild.me)
        if not perms.read_messages or not perms.manage_messages:
            continue
        try:
            async for msg in ch.history(limit=SCAN_LIMIT, oldest_first=False):
                if msg.author.bot:
                    continue
                content = msg.content or ""
                # quick heuristic
                flags = heuristic_text_flags(content)
                # external moderation can augment flags
                mod = {}
                try:
                    mod = await external_moderation_check(content)
                except Exception:
                    mod = {}
                delete_msg = False
                reason = None
                if flags.get("toxic") or flags.get("controversial"):
                    delete_msg = True
                    reason = "heuristic_toxic_or_controversial"
                if flags.get("suspicious_code"):
                    delete_msg = True
                    reason = f"suspicious_code:{flags.get('suspicious_code')}"
                # external moderation result (example keys: toxic, nsfw)
                if isinstance(mod, dict):
                    if mod.get("toxic") is True or mod.get("nsfw") is True or mod.get("flagged") is True:
                        delete_msg = True
                        reason = f"external_mod:{mod}"
                if delete_msg:
                    try:
                        await msg.delete()
                        REPORT["toxic_messages"].append({"channel": ch.name, "author": str(msg.author), "snippet": content[:220], "reason": reason, "ts": now_iso()})
                    except Exception as e:
                        REPORT["errors"].append(f"delete_msg_error:{str(e)}")
                    continue

                # attachments: light checks
                for att in msg.attachments:
                    try:
                        if any(att.filename.lower().endswith(ext) for ext in SCAN_IMAGE_EXTS) and filename_suspicious(att.filename):
                            await msg.delete()
                            REPORT["toxic_messages"].append({"channel": ch.name, "author": str(msg.author), "attachment": att.url, "reason":"suspicious_filename", "ts": now_iso()})
                            break
                        # optionally fetch and send to moderation API (skipped by default for speed)
                    except Exception:
                        continue
        except Exception as e:
            REPORT["errors"].append(f"scan_channel_error_{ch.name}:{str(e)}")

# ---------------- NEWS MERGE & SUMMARIZE ----------------
# fetch multiple sources (lightweight)
import xml.etree.ElementTree as ET

async def fetch_reddit(subreddit="worldnews", limit=12, mode="hot"):
    url = f"https://www.reddit.com/r/{subreddit}/{mode}.json?limit={limit}"
    headers = {"User-Agent": "DiscordNewsBot/1.0"}
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, headers=headers, timeout=10) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                return [p["data"]["title"] for p in data.get("data", {}).get("children", []) if p.get("data",{}).get("title")]
    except Exception as e:
        REPORT["errors"].append(f"reddit_fetch_err:{str(e)}")
        return []

async def fetch_rss(url, max_items=8):
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=10) as r:
                if r.status != 200:
                    return []
                txt = await r.text()
                # parse RSS quickly
                titles = re.findall(r"<title>(.*?)</title>", txt, flags=re.IGNORECASE|re.DOTALL)
                # skip first if it's channel title
                if len(titles) > 1:
                    titles = titles[1:1+max_items]
                return [re.sub(r"<.*?>","",t).strip() for t in titles]
    except Exception as e:
        REPORT["errors"].append(f"rss_fetch_err:{str(e)}")
        return []

def dedupe_similar(headlines):
    unique = []
    seen = set()
    for s in headlines:
        norm = re.sub(r"\W+"," ", s.lower()).strip()
        key = " ".join(norm.split()[:8])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique

async def build_news_summary():
    # concurrent fetch
    tasks = [
        fetch_reddit("worldnews", limit=10, mode=random.choice(["hot","new"])),
        fetch_reddit("news", limit=8, mode=random.choice(["hot","new"])),
        fetch_rss("http://feeds.bbci.co.uk/news/world/rss.xml", max_items=6),
        fetch_rss("https://feeds.apnews.com/apf-topnews", max_items=6)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    headlines = []
    for r in results:
        if isinstance(r, Exception):
            continue
        headlines.extend(r or [])
    headlines = [h for h in headlines if h and len(h)>12][:60]
    headlines = dedupe_similar(headlines)
    if not headlines:
        return "Unable to fetch news today."
    # merge into a single text for AI
    merged = "\n".join(["- " + h for h in headlines[:30]])
    prompt = f"""
    Merge and neutralize the following headlines into a short unbiased daily news briefing.
    Keep it concise (2-6 sentences), avoid sensational language, and mention no single source preference.
    Headlines:
    {merged}
    """
    summary = await ai_call(prompt, max_tokens=220)
    return summary

# ---------------- FUN DAILY: CHAT SUMMARY + STATS ----------------
async def summarize_chat_history(guild: discord.Guild):
    """
    Read last N messages per channel (bounded) and produce:
    - fun summary
    - server stats (most talkative, total messages)
    - server mood score
    """
    messages = []
    per_user = Counter()
    per_channel = Counter()
    earliest_ts = None

    # iterate channels (only text)
    for ch in guild.text_channels:
        perms = ch.permissions_for(guild.me)
        if not perms.read_messages:
            continue
        try:
            async for msg in ch.history(limit=MAX_HISTORY_MESSAGES, oldest_first=False):
                if msg.author.bot:
                    continue
                content = (msg.content or "").strip()
                if not content:
                    continue
                messages.append(f"{msg.author.display_name}: {content}")
                per_user[msg.author.display_name] += 1
                per_channel[ch.name] += 1
        except Exception:
            continue

    total_msgs = sum(per_user.values())
    if total_msgs == 0:
        return ("No chat activity found today.", {})

    # build prompt for AI summarization
    sample_text = "\n".join(messages[:1000])  # cap
    prompt = f"""
    You are a playful Discord reporter. Summarize the following chat in a FUN tone.
    Provide:
    1) A short gossip-style headline (one line)
    2) 2-4 sentence comedic recap
    3) Top 3 funniest moments (bullet list)
    4) A Server Mood score (0-100) and a one-line explanation
    Chat excerpts:
    {sample_text}
    """
    ai_summary = await ai_call(prompt, max_tokens=300)
    stats = {
        "total_messages": total_msgs,
        "top_users": per_user.most_common(5),
        "top_channels": per_channel.most_common(5)
    }
    # store in report
    REPORT["daily_summary"] = {"ai_summary_snippet": ai_summary[:300], "stats": stats, "ts": now_iso()}
    return ai_summary, stats

# ---------------- POST DAILY (main orchestration) ----------------
async def post_daily_reports(guild: discord.Guild):
    # ensure daily-info exists
    channel = get(guild.text_channels, name="daily-info") or await guild.create_text_channel("daily-info", topic="AI-generated daily news & fun summary")

    # news
    news = await build_news_summary()
    try:
        await channel.send("📰 **AI Daily News Summary (merged & neutral)**")
        await channel.send(news)
    except Exception:
        REPORT["errors"].append("failed_post_news")

    # fun chat summary + stats
    chat_summary, stats = await summarize_chat_history(guild)
    try:
        await channel.send("🎭 **Fun Chat Summary**")
        await channel.send(chat_summary)
        # stats embed (concise)
        stats_text = f"• Total messages: {stats['total_messages']}\n• Top users: " + ", ".join([f"{u} ({c})" for u,c in stats["top_users"][:3]])
        await channel.send(stats_text)
    except Exception:
        REPORT["errors"].append("failed_post_chat_summary")

    # meme & fortune
    meme = await ai_call("Generate a single short funny 'meme line of the day' (10 words max).", max_tokens=40)
    fortune = await ai_call("Generate 3 short fortune cookie lines (one wholesome, one chaotic, one oddly specific). Format as bullets.", max_tokens=120)
    lucky = random.randint(1, 9999)
    mood = await ai_call("In one short sentence, rate the server mood today on scale 0-100 and give one-line reason.", max_tokens=40)

    try:
        await channel.send("🤡 **Meme Line of the Day**")
        await channel.send(meme if meme else random.choice(FUN_MEMES))
        await channel.send("🔮 **Fortune Cookies**")
        await channel.send(fortune if fortune else "\n".join(random.sample(FORTUNES, 3)))
        await channel.send(f"🍀 **Daily Lucky Number:** `{lucky}`")
        await channel.send(f"📊 **Server Mood:** {mood}")
    except Exception:
        REPORT["errors"].append("failed_post_fun_stuff")

# ---------------- MAIN RUN ----------------
@client.event
async def on_ready():
    print(f"[{now_iso()}] Bot ready as {client.user}. Starting daily tasks.")
    # minimal checks
    if not DISCORD_TOKEN or not GUILD_ID:
        print("DISCORD_TOKEN or GUILD_ID missing.", file=sys.stderr)
        await client.close()
        return

    guild = client.get_guild(GUILD_ID)
    if not guild:
        print("Guild not found for GUILD_ID:", GUILD_ID, file=sys.stderr)
        await client.close()
        return

    # load trusted snapshot
    prev_snap = load_json(DEFAULT_CHANNELS_FILE + ".snapshot") or {}
    # detect nuke
    nuke_info = detect_nuke(prev_snap, guild)
    REPORT["nuke_events"].append(nuke_info)
    if nuke_info.get("is_nuke"):
        deleted = await remediate_foreign_channels(guild, prev_snap)
        REPORT["deleted_channels"].extend(deleted)
        REPORT["nuke_events"][-1]["deleted_channels"] = deleted

    # scan messages and clean
    await scan_and_clean_messages(guild)

    # post daily aggregated reports
    await post_daily_reports(guild)

    # save snapshot for next run
    snap = build_snapshot(guild)
    save_json(DEFAULT_CHANNELS_FILE + ".snapshot", snap)
    # save REPORT
    save_json(REPORT_FILE, REPORT)

    print(f"[{now_iso()}] Daily run complete. Exiting.")
    await client.close()
    # do not attempt to sys.exit — let environment stop the process

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
