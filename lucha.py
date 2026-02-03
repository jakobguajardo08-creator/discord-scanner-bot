#!/usr/bin/env python3
"""
LuCha — Daily Channel Security Scan

- Runs once per day
- Restores missing channels from channels.json
- Deletes channels not listed in channels.json
- Exits immediately after run
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timezone, date
from typing import Dict, Set, Optional

import discord
from discord.utils import get

# ---------------- Config ----------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GUILD_ID = int(os.getenv("GUILD_ID", "0"))

CHANNELS_FILE = "channels.json"
STATE_FILE = "lucha_state.json"
REPORT_FILE = "mod_report.json"

LUCHA_ARMED = os.getenv("LUCHA_ARMED", "0") == "1"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1" or not LUCHA_ARMED

if not DISCORD_TOKEN or not GUILD_ID:
    print("DISCORD_TOKEN and GUILD_ID are required")
    sys.exit(1)

# ---------------- Helpers ----------------
def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def today_key() -> str:
    return f"ran:{date.today().isoformat()}"

# ---------------- Discord ----------------
intents = discord.Intents.default()
intents.guilds = True
client = discord.Client(intents=intents)

REPORT = {
    "ts": None,
    "dry_run": DRY_RUN,
    "actions": [],
    "errors": []
}

@client.event
async def on_ready():
    REPORT["ts"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    # ---- once per day lock ----
    state = load_json(STATE_FILE)
    if state.get(today_key()):
        REPORT["actions"].append("exit_already_ran_today")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    guild = client.get_guild(GUILD_ID)
    if not guild:
        REPORT["errors"].append("guild_not_found")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    spec = load_json(CHANNELS_FILE)
    desired = spec.get("channels", [])

    # ---- build desired keys ----
    def key(t, n):
        return f"{t}:{n.lower()}"

    desired_keys: Set[str] = set()
    for ch in desired:
        desired_keys.add(key(ch["type"], ch["name"]))

    # ---- system channels safety ----
    system_ids = set()
    for attr in (
        "system_channel",
        "rules_channel",
        "public_updates_channel",
        "safety_alerts_channel",
    ):
        c = getattr(guild, attr, None)
        if c:
            system_ids.add(c.id)

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_manage_channels_permission")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- restore missing channels ----
    for item in desired:
        t = item["type"]
        name = item["name"]
        cat_name = item.get("category")

        exists = False
        for ch in guild.channels:
            if ch.name.lower() == name.lower():
                exists = True
                break

        if exists:
            continue

        if DRY_RUN:
            REPORT["actions"].append(f"would_create_{t}:{name}")
            continue

        category = get(guild.categories, name=cat_name) if cat_name else None

        try:
            if t == "category":
                await guild.create_category(name)
            elif t == "text":
                await guild.create_text_channel(name, category=category)
            elif t == "voice":
                await guild.create_voice_channel(name, category=category)

            REPORT["actions"].append(f"create_{t}:{name}")
            await asyncio.sleep(1.2)
        except Exception as e:
            REPORT["errors"].append(f"create_error:{name}:{e}")

    # ---- delete foreign channels ----
    for ch in list(guild.channels):
        if ch.id in system_ids:
            continue

        if not isinstance(ch, (discord.TextChannel, discord.VoiceChannel, discord.CategoryChannel)):
            continue

        ch_type = (
            "category" if isinstance(ch, discord.CategoryChannel)
            else "text" if isinstance(ch, discord.TextChannel)
            else "voice"
        )

        k = key(ch_type, ch.name)
        if k in desired_keys:
            continue

        if DRY_RUN:
            REPORT["actions"].append(f"would_delete:{ch.name}")
            continue

        try:
            await ch.delete(reason="LuCha daily channel reconciliation")
            REPORT["actions"].append(f"delete:{ch.name}")
            await asyncio.sleep(1.2)
        except Exception as e:
            REPORT["errors"].append(f"delete_error:{ch.name}:{e}")

    # ---- finalize ----
    state[today_key()] = True
    save_json(STATE_FILE, state)
    save_json(REPORT_FILE, REPORT)

    await client.close()

client.run(DISCORD_TOKEN)
