#!/usr/bin/env python3
"""
daily_scan_rewrite.py

Rewritten Discord moderation & daily-news bot:
- Multi-source headline fetch (Reddit + RSS)
- Embedding-based clustering & neutral merging of related headlines
- Summarization of merged clusters to produce a single unbiased briefing
- Conservative destructive actions:
    * delete messages (toxic/NSFW/credential leaks/malicious code)
    * delete channels/categories that are clearly foreign after a nuke
  (no daily blanket delete -> only removes things that don't belong)
- Snapshot/backup of roles & channels used to identify "trusted" state
- REPORT JSON output of actions taken

CONFIG via environment variables:
- DISCORD_TOKEN, GUILD_ID
- BACKUP_FILE (optional) default "guild_backup.json"
- DEFAULT_CHANNELS_FILE (optional) default "default_channels.json"
"""

import os
import sys
import json
import io
import asyncio
import random
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple
import logging
import math

import aiohttp
import discord
from discord.utils import get

# Optional ML dependencies (try to import; fallback gracefully)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoFeatureExtractor, AutoModelForImageClassification
    import torch
    TF_AVAILABLE = False
except Exception:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoFeatureExtractor = None
    AutoModelForImageClassification = None
    torch = None
    TF_AVAILABLE = False

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    ST_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    np = None
    ST_AVAILABLE = False

# ---------------- CONFIG ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = int(os.getenv("GUILD_ID")) if os.getenv("GUILD_ID") else None
BACKUP_FILE = os.getenv("BACKUP_FILE", "guild_backup.json")
DEFAULT_CHANNELS_FILE = os.getenv("DEFAULT_CHANNELS_FILE", "default_channels.json")
REPORT_FILE = os.getenv("REPORT_FILE", "mod_report.json")

# Tunables
MAX_MESSAGES = int(os.getenv("MAX_MESSAGES", "300"))
TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.6"))
NSFW_THRESHOLD = float(os.getenv("NSFW_THRESHOLD", "0.75"))
SUSPICIOUS_PATTERNS = [
    r"(api_key|secret|token)\s*[:=]\s*[A-Za-z0-9_\-]{8,}",
    r"exec\(",
    r"eval\(",
    r"base64\.b64decode",
    r"curl .* --output",
    r"powershell .* -EncodedCommand",
    r"wget .* -O",
]
NUKE_DELETION_THRESHOLD = float(os.getenv("NUKE_DELETION_THRESHOLD", "0.15"))  # fraction of channels deleted to consider a nuke
NUKE_NEW_CHANNEL_THRESHOLD = int(os.getenv("NUKE_NEW_CHANNEL_THRESHOLD", "6"))  # new channels created to consider an attack
SIMILARITY_MERGE_THRESHOLD = float(os.getenv("SIMILARITY_MERGE_THRESHOLD", "0.72"))  # for headline merging
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
SUMMARY_MODEL_NAME = os.getenv("SUMMARY_MODEL_NAME", "facebook/bart-large-cnn")

SCAN_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

SEASONAL_EMOJIS = {"🎃": [10, 11], "🎄": [12], "💖": [2], "🌸": [3, 4]}

# If user wants "delete anything controversial" — we'll treat controversial as:
# toxic content, NSFW content, credential leaks, malicious code, phishing links, extremist slurs.
# Politically charged discussion: we treat "politics" as controversial only if explicit insults/threats happen.

REPORT = {
    "toxic_messages": [],
    "nsfw_attachments": [],
    "suspicious_code": [],
    "nuke_events": [],
    "restoration_actions": [],
    "news_summary": {},
    "decorations": [],
    "errors": []
}

# Check config
if not DISCORD_TOKEN or not GUILD_ID:
    print("DISCORD_TOKEN or GUILD_ID not set", file=sys.stderr)
    sys.exit(2)

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("daily_scan_rewrite")

# ---------------- ML model setup (graceful) ----------------
txt_model = None
txt_tokenizer = None
img_extractor = None
img_model = None
summarizer = None
embed_model = None

def init_models():
    global txt_model, txt_tokenizer, img_extractor, img_model, summarizer, embed_model
    # Text toxicity model
    try:
        if AutoTokenizer and AutoModelForSequenceClassification:
            txt_tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")
            txt_model = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")
            log.info("Loaded toxic-bert model.")
    except Exception as e:
        log.warning("Could not load toxic text model: %s", e)
        REPORT["errors"].append(f"toxic_model_load_error: {str(e)}")

    # Image NSFW detection
    try:
        if AutoFeatureExtractor and AutoModelForImageClassification:
            img_extractor = AutoFeatureExtractor.from_pretrained("Falconsai/nsfw_image_detection")
            img_model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
            log.info("Loaded nsfw image model.")
    except Exception as e:
        log.warning("Could not load nsfw image model: %s", e)
        REPORT["errors"].append(f"nsfw_model_load_error: {str(e)}")

    # Summarizer
    try:
        if pipeline:
            summarizer = pipeline("summarization", model=SUMMARY_MODEL_NAME, framework="pt")
            log.info("Loaded summarizer pipeline.")
    except Exception as e:
        log.warning("Could not load summarizer: %s", e)
        REPORT["errors"].append(f"summarizer_load_error: {str(e)}")

    # Embedding model (sentence-transformers)
    try:
        if ST_AVAILABLE:
            embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            log.info("Loaded sentence-transformers embedding model.")
    except Exception as e:
        log.warning("Could not load embedding model: %s", e)
        REPORT["errors"].append(f"embedding_model_load_error: {str(e)}")

# ---------------- UTILITIES ----------------
def load_json(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log.warning("Failed loading json %s: %s", path, e)
    return {}

def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning("Failed saving json %s: %s", path, e)

def get_dynamic_emoji():
    month = datetime.utcnow().month
    for e, months in SEASONAL_EMOJIS.items():
        if month in months:
            return e
    return "🔹"

def suspicious_code_check(text: str):
    for pat in SUSPICIOUS_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return pat
    return None

# ---------------- embeddings + clustering helpers ----------------
def cosine_sim(a, b):
    # expects 1D numpy arrays
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def cluster_headlines(headlines: List[str], threshold: float = SIMILARITY_MERGE_THRESHOLD) -> List[List[int]]:
    """
    Simple greedy clustering:
    - compute embeddings
    - pick an unclustered headline, create a cluster, add any headline with similarity >= threshold
    - continue
    Returns list of clusters expressed as index lists
    """
    if not ST_AVAILABLE or not embed_model:
        # fallback: topic naive grouping by shared words
        clusters = []
        assigned = set()
        for i, h in enumerate(headlines):
            if i in assigned: continue
            cluster = [i]
            assigned.add(i)
            words = set(re.findall(r"\w{4,}", h.lower()))
            for j in range(i+1, len(headlines)):
                if j in assigned: continue
                words_j = set(re.findall(r"\w{4,}", headlines[j].lower()))
                if len(words & words_j) >= 2:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)
        return clusters

    # compute embeddings
    texts = headlines
    try:
        embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    except Exception:
        # fallback naive
        return cluster_headlines(headlines, threshold)

    n = len(texts)
    clustered = [False] * n
    clusters = []
    for i in range(n):
        if clustered[i]: continue
        cluster = [i]
        clustered[i] = True
        for j in range(i+1, n):
            if clustered[j]: continue
            s = cosine_sim(embeddings[i], embeddings[j])
            if s >= threshold:
                cluster.append(j)
                clustered[j] = True
        clusters.append(cluster)
    return clusters

def merge_cluster_texts(headlines: List[str], cluster: List[int]) -> str:
    # simple merge: choose shortest balanced header then append unique fragments
    pieces = [headlines[i] for i in cluster]
    # pick a representative: shortest non-truncated
    rep = min(pieces, key=lambda s: len(s))
    # append additional unique headlines as bullets for context
    extras = [p for p in pieces if p != rep]
    merged = rep
    if extras:
        merged += "\n" + "\n".join(f"- {e}" for e in extras[:5])
    return merged

# ---------------- NEWS FETCHING ----------------
# We'll fetch from Reddit (hot/new) and a few RSS sources. Use aiohttp for async requests.
RSS_SOURCES = [
    ("Reuters", "https://www.reuters.com/tools/rss"),
    ("AP", "https://apnews.com/hub/ap-top-news?format=rss"),
    ("BBC", "http://feeds.bbci.co.uk/news/world/rss.xml"),
]

REDDIT_SUBREDDITS = ["worldnews", "news", "politics"]

async def fetch_reddit_headlines(session: aiohttp.ClientSession, subreddit: str, limit: int = 20, mode: str = "hot"):
    url = f"https://www.reddit.com/r/{subreddit}/{mode}.json?limit={limit}"
    headers = {"User-Agent": "DiscordBot-NewsFetcher/2.0"}
    try:
        async with session.get(url, headers=headers, timeout=15) as r:
            data = await r.json()
            posts = []
            for p in data.get("data", {}).get("children", []):
                title = p["data"].get("title")
                if title:
                    posts.append(title)
            return posts
    except Exception as e:
        REPORT["errors"].append(f"reddit_fetch_error_{subreddit}:{str(e)}")
        return []

async def fetch_rss_headlines(session: aiohttp.ClientSession, url: str, max_items: int = 12):
    try:
        async with session.get(url, timeout=15) as r:
            text = await r.text()
            # naive parse of <title> elements (skip channel title)
            titles = re.findall(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
            # first <title> is usually the feed title; skip it
            if len(titles) > 1:
                titles = titles[1:1 + max_items]
            else:
                titles = titles[:max_items]
            # strip tags inside titles
            clean = [re.sub(r"<.*?>", "", t).strip() for t in titles]
            return clean
    except Exception as e:
        REPORT["errors"].append(f"rss_fetch_error_{url}:{str(e)}")
        return []

async def gather_headlines():
    async with aiohttp.ClientSession() as session:
        tasks = []
        # reddit
        for subreddit in REDDIT_SUBREDDITS:
            limit = random.randint(8, 25)
            mode = random.choice(["hot", "new"])
            tasks.append(fetch_reddit_headlines(session, subreddit, limit=limit, mode=mode))
        # rss
        for name, url in RSS_SOURCES:
            tasks.append(fetch_rss_headlines(session, url, max_items=12))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # flatten and unique preserving order
        headlines = []
        seen = set()
        for r in results:
            if isinstance(r, Exception): continue
            for h in (r or []):
                h = h.strip()
                if not h: continue
                if h not in seen:
                    seen.add(h)
                    headlines.append(h)
        # shuffle a little to avoid identical ordering
        random.shuffle(headlines)
        return headlines[:60]  # cap

# ---------------- SUMMARIZATION ----------------
def summarize_text(text: str, max_length: int = 180, min_length: int = 40):
    if summarizer:
        try:
            # Use sampling to add variety but keep it deterministic-ish
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=True, top_p=0.9, temperature=0.6)[0]["summary_text"]
            return summary
        except Exception as e:
            REPORT["errors"].append(f"summarizer_runtime_error:{str(e)}")
            return text
    # fallback: naive truncation/cleaning
    lines = text.splitlines()
    short = " ".join(lines[:6])
    return (short[:max_length] + "...") if len(short) > max_length else short

async def build_daily_briefing():
    headlines = await gather_headlines()
    if not headlines:
        return "Unable to fetch headlines today."

    clusters = cluster_headlines(headlines, threshold=SIMILARITY_MERGE_THRESHOLD)
    merged_texts = [merge_cluster_texts(headlines, c) for c in clusters]

    # Build neutral merged text: join cluster reps separated by blank lines
    combined = "\n\n".join(merged_texts)
    # Summarize combined into one short briefing
    summary = summarize_text("Today's top stories:\n\n" + combined, max_length=200, min_length=60)
    # fill REPORT for trace
    REPORT["news_summary"] = {
        "headlines_count": len(headlines),
        "clusters": len(clusters),
        "merged_count": len(merged_texts),
        "summary_snippet": summary[:300]
    }
    return summary

# ---------------- NSFW IMAGE SCAN ----------------
def run_model_on_image_bytes(data: bytes) -> float:
    if not img_extractor or not img_model:
        return 0.0
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGB")
        inputs = img_extractor(images=img, return_tensors="pt")
        outputs = img_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].detach().cpu().numpy()
        # assume index 1 is nsfw positive
        if len(probs) > 1:
            return float(probs[1])
        return float(max(probs))
    except Exception as e:
        REPORT["errors"].append(f"image_scan_error:{str(e)}")
        return 0.0

# ---------------- TEXT TOXICITY SCAN ----------------
def run_model_on_text(text: str) -> Dict[str, float]:
    if not txt_tokenizer or not txt_model:
        # fallback heuristic: presence of harsh words
        heur = {}
        lower = text.lower()
        bad = ["idiot", "kill", "stfu", "die", "racist", "nazi", "faggot", "retard"]
        score = 0.0
        for b in bad:
            if b in lower:
                score += 0.25
        heur["toxic"] = min(score, 1.0)
        return heur
    try:
        inputs = txt_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        outputs = txt_model(**inputs)
        probs = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy()
        # some models have different label order; we map only the first 'toxic' score if present
        return {"toxic": float(probs[0])}
    except Exception as e:
        REPORT["errors"].append(f"text_scan_error:{str(e)}")
        return {"toxic": 0.0}

def is_text_toxic(text: str) -> Tuple[bool, Dict[str, float]]:
    res = run_model_on_text(text)
    score = float(res.get("toxic", 0.0))
    return (score >= TOXIC_THRESHOLD, res)

# ---------------- SNIPPET CLEANUP ----------------
def short_snip(s: str, n=300):
    return (s[:n] + "...") if len(s) > n else s

# ---------------- NATIVE GUILD SNAPSHOT & NUKE DETECTION ----------------
def build_snapshot(guild: discord.Guild) -> dict:
    snapshot = {"roles": [], "channels": [], "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()}
    for role in guild.roles:
        snapshot["roles"].append({"name": role.name, "permissions": role.permissions.value, "hoist": role.hoist, "mentionable": role.mentionable})
    for ch in guild.channels:
        ch_info = {"id": ch.id, "name": ch.name, "type": str(ch.type), "position": ch.position}
        if isinstance(ch, discord.TextChannel):
            ch_info.update({"topic": ch.topic, "nsfw": ch.nsfw, "slowmode_delay": ch.slowmode_delay})
        snapshot["channels"].append(ch_info)
    return snapshot

def detect_nuke_from_snap(prev_snapshot: dict, guild: discord.Guild) -> dict:
    """
    Decide whether a nuke/attack happened by comparing prev channels to current.
    If many trusted channels are missing and many new ones exist -> nuke.
    """
    prev_channels = prev_snapshot.get("channels", []) if prev_snapshot else []
    prev_names = [c.get("name") for c in prev_channels]
    curr_channels = list(guild.channels)
    curr_names = [c.name for c in curr_channels]

    missing = [n for n in prev_names if n not in curr_names]
    new = [c for c in curr_channels if c.name not in prev_names]

    result = {
        "deleted_count": len(missing),
        "created_count": len(new),
        "missing_sample": missing[:10],
        "new_sample": [c.name for c in new[:10]]
    }

    # if deleted fraction high and many new channels created -> nuke
    if len(prev_names) > 0 and (len(missing) > max(1, int(len(prev_names) * NUKE_DELETION_THRESHOLD)) and len(new) >= NUKE_NEW_CHANNEL_THRESHOLD):
        result["is_nuke"] = True
    else:
        result["is_nuke"] = False
    return result

async def remediate_nuke(guild: discord.Guild, prev_snapshot: dict):
    # Remove channels/categories that clearly don't belong: channels not present in prev snapshot and created recently.
    # We err on side of caution: only delete if channel is new and name contains suspicious tokens or channel count exploded.
    prev_names = {c.get("name") for c in (prev_snapshot.get("channels", []) if prev_snapshot else [])}
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    deleted_channels = []
    for ch in guild.channels:
        if ch.name in prev_names:
            continue
        # channel not in snapshot -> candidate for removal
        # Heuristics: if name contains suspicious tokens or server was nuked
        suspect_name = bool(re.search(r"(spam|raid|hacked|nuke|free-giveaway|giveaway|bot-)", ch.name, flags=re.IGNORECASE))
        # or newly created within past 24 hours (discord doesn't give created timestamp easily for channels; rely on position heuristic)
        # We'll check permissions: if channel has no overwrites for admins and very open -> suspicious
        try:
            overwrites = ch.overwrites
        except Exception:
            overwrites = {}
        if suspect_name or len(guild.channels) > (len(prev_names) + 8):
            try:
                await ch.delete(reason="Detected foreign channel after nuke/attack")
                deleted_channels.append(ch.name)
                REPORT["restoration_actions"].append({"action": "delete_channel", "channel": ch.name, "timestamp": datetime.utcnow().isoformat()})
            except Exception as e:
                REPORT["errors"].append(f"failed_delete_channel_{ch.name}:{str(e)}")
    return deleted_channels

# ---------------- MESSAGE SCAN ----------------
async def scan_messages_and_cleanup(guild: discord.Guild):
    """
    Scan recent messages in text channels and:
    - delete messages that are toxic (by model or heuristics)
    - delete messages with NSFW attachments above threshold
    - delete messages that contain suspicious code/credentials
    Only perform deletion when confidence is high.
    """
    for ch in guild.text_channels:
        perms = ch.permissions_for(guild.me)
        if not perms.read_messages or not perms.read_message_history or not perms.manage_messages:
            continue
        try:
            async for msg in ch.history(limit=MAX_MESSAGES, oldest_first=False):
                if msg.author.bot:
                    continue
                # check text
                if msg.content:
                    toxic, scores = is_text_toxic(msg.content)
                    if toxic:
                        REPORT["toxic_messages"].append({
                            "channel": ch.name,
                            "author": str(msg.author),
                            "snippet": short_snip(msg.content),
                            "scores": scores,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        try:
                            await msg.delete(reason="Toxic content (automated moderation)")
                        except Exception as e:
                            REPORT["errors"].append(f"delete_msg_error:{str(e)}")

                    pat = suspicious_code_check(msg.content)
                    if pat:
                        REPORT["suspicious_code"].append({
                            "channel": ch.name,
                            "author": str(msg.author),
                            "pattern": pat,
                            "snippet": short_snip(msg.content),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        # Delete if pattern indicates leaked credential or exec attempt
                        try:
                            await msg.delete(reason=f"Suspicious code/content matched pattern: {pat}")
                        except Exception as e:
                            REPORT["errors"].append(f"delete_msg_error:{str(e)}")

                # check attachments
                for att in msg.attachments:
                    if any(att.filename.lower().endswith(ext) for ext in SCAN_IMAGE_TYPES):
                        data = None
                        try:
                            data = await att.read()
                        except Exception:
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(att.url, timeout=10) as r:
                                        if r.status == 200:
                                            data = await r.read()
                            except Exception:
                                data = None
                        if data:
                            nsfw_score = run_model_on_image_bytes(data)
                            if nsfw_score >= NSFW_THRESHOLD:
                                REPORT["nsfw_attachments"].append({
                                    "channel": ch.name,
                                    "author": str(msg.author),
                                    "attachment": att.url,
                                    "score": nsfw_score,
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                try:
                                    await msg.delete(reason="NSFW attachment (automated moderation)")
                                except Exception as e:
                                    REPORT["errors"].append(f"delete_msg_error:{str(e)}")
        except Exception as e:
            REPORT["errors"].append(f"scan_channel_error_{ch.name}:{str(e)}")

# ---------------- CHANNEL RESTORATION (NON-DESTRUCTIVE) ----------------
async def ensure_daily_info_channel(guild: discord.Guild):
    channel = get(guild.text_channels, name="daily-info")
    if channel:
        return channel
    try:
        channel = await guild.create_text_channel("daily-info", topic="AI-generated daily news summary")
        REPORT["restoration_actions"].append({"action": "create_channel", "channel": "daily-info", "timestamp": datetime.utcnow().isoformat()})
        return channel
    except Exception as e:
        REPORT["errors"].append(f"create_daily_channel_error:{str(e)}")
        return None

# ---------------- MAIN DISCORD CLIENT ----------------
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    log.info("Bot ready. Initializing models & running daily tasks.")
    init_models()
    try:
        guild = client.get_guild(GUILD_ID)
        if not guild:
            log.error("Guild not found (GUILD_ID=%s).", GUILD_ID)
            await client.close()
            return

        # Load trusted snapshot
        prev_snapshot = load_json(BACKUP_FILE) or {}
        # Detect nuke
        nuke_info = detect_nuke_from_snap(prev_snapshot, guild)
        REPORT["nuke_events"].append(nuke_info)
        if nuke_info.get("is_nuke"):
            log.warning("Nuke detected: %s", nuke_info)
            # Remediate suspicious channels
            deleted = await remediate_nuke(guild, prev_snapshot)
            REPORT["nuke_events"][-1]["deleted_channels"] = deleted

        # Scan messages & perform conservative deletion (toxicity, nsfw, suspicious code)
        await scan_messages_and_cleanup(guild)

        # Build and post daily briefing
        channel = await ensure_daily_info_channel(guild)
        summary = await build_daily_briefing()
        if channel:
            try:
                emoji = get_dynamic_emoji()
                await channel.send(f"{emoji} **AI Daily News Summary (merged & neutralized)**")
                # Break summary into chunks if too long
                for chunk in [summary[i:i+1900] for i in range(0, len(summary), 1900)]:
                    await channel.send(chunk)
                REPORT["restoration_actions"].append({"action": "post_daily_summary", "channel": channel.name, "timestamp": datetime.utcnow().isoformat()})
            except Exception as e:
                REPORT["errors"].append(f"post_daily_summary_error:{str(e)}")

        # Save new snapshot for future comparisons
        snapshot = build_snapshot(guild)
        save_json(BACKUP_FILE, snapshot)
        # save report
        save_json(REPORT_FILE, REPORT)
        log.info("Daily tasks completed. Exiting.")
    except Exception as e:
        log.exception("Unhandled error in on_ready: %s", e)
        REPORT["errors"].append(f"on_ready_unhandled:{str(e)}")
        save_json(REPORT_FILE, REPORT)
    finally:
        # close client gracefully
        try:
            await client.close()
        finally:
            # try to exit the process
            try:
                sys.exit(0)
            except SystemExit:
                pass

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
