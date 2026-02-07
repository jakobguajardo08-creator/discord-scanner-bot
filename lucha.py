#!/usr/bin/env python3
"""
LuCha — SAFE Daily Channel Reconciler

ABSOLUTE SAFETY GUARANTEES:
• Will NOT run if channels.json is missing or empty
• Will NOT delete unless creation succeeds
• Will NOT touch Discord system channels
• Will run once per day unless FORCE_RUN=1
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timezone, date
from typing import Dict, Set

import discord
from discord.utils import get

# ================= CONFIG =================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GUILD_ID = int(os.getenv("GUILD_ID", "0"))

CHANNELS_FILE = "channels.json"
STATE_FILE = "lucha_state.json"
REPORT_FILE = "mod_report.json"

LUCHA_ARMED = os.getenv("LUCHA_ARMED", "0") == "1"
FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1" or not LUCHA_ARMED

if not DISCORD_TOKEN or not GUILD_ID:
    print("❌ Missing DISCORD_TOKEN or GUILD_ID")
    sys.exit(1)

# ================= HELPERS =================
def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

def today_key():
    return f"ran:{date.today().isoformat()}"

# ================= DISCORD =================
intents = discord.Intents.default()
intents.guilds = True
client = discord.Client(intents=intents)

REPORT = {
    "timestamp": None,
    "dry_run": DRY_RUN,
    "actions": [],
    "errors": []
}

# ================= PERMISSIONS =================
PERM_MAP = {
    "view_channel": "view_channel",
    "send_messages": "send_messages",
    "add_reactions": "add_reactions",
    "manage_messages": "manage_messages",
    "connect": "connect",
    "speak": "speak",
}

def build_overwrites(guild, perm_spec):
    overwrites = {}
    everyone = guild.default_role

    if "everyone" in perm_spec:
        p = discord.PermissionOverwrite()
        for k, v in perm_spec["everyone"].items():
            setattr(p, PERM_MAP[k], v)
        overwrites[everyone] = p

    for role_name, rules in perm_spec.get("roles", {}).items():
        role = get(guild.roles, name=role_name)
        if not role:
            continue
        p = discord.PermissionOverwrite()
        for k, v in rules.items():
            setattr(p, PERM_MAP[k], v)
        overwrites[role] = p

    return overwrites

# ================= BOT =================
@client.event
async def on_ready():
    REPORT["timestamp"] = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    # ---- DAILY LOCK ----
    state = load_json(STATE_FILE)
    if state.get(today_key()) and not FORCE_RUN:
        REPORT["actions"].append("exit_already_ran_today")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- LOAD & VALIDATE JSON ----
    spec_data = load_json(CHANNELS_FILE)
    spec = spec_data.get("channels")

    if not isinstance(spec, list) or len(spec) == 0:
        REPORT["errors"].append(
            f"ABORTED: {CHANNELS_FILE} missing or empty"
        )
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    guild = client.get_guild(GUILD_ID)
    if not guild:
        REPORT["errors"].append("guild_not_found")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_manage_channels_permission")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- SYSTEM CHANNEL SAFETY ----
    system_ids = set()
    for attr in (
        "system_channel",
        "rules_channel",
        "public_updates_channel",
        "safety_alerts_channel",
    ):
        ch = getattr(guild, attr, None)
        if ch:
            system_ids.add(ch.id)

    # ---- STEP 1: CREATE CATEGORIES ----
    category_map = {}
    creation_failed = False

    for c in spec:
        if c["type"] != "category":
            continue

        cat = get(guild.categories, name=c["name"])
        if cat:
            category_map[c["name"]] = cat
            continue

        if DRY_RUN:
            REPORT["actions"].append(f"would_create_category:{c['name']}")
            continue

        try:
            cat = await guild.create_category(c["name"])
            category_map[c["name"]] = cat
            REPORT["actions"].append(f"create_category:{c['name']}")
            await asyncio.sleep(1.2)
        except Exception as e:
            creation_failed = True
            REPORT["errors"].append(f"category_create_error:{c['name']}:{e}")

    # ---- STEP 2: CREATE CHANNELS ----
    for c in spec:
        if c["type"] == "category":
            continue

        if any(ch.name.lower() == c["name"].lower() for ch in guild.channels):
            continue

        parent = category_map.get(c.get("category"))
        overwrites = build_overwrites(guild, c.get("permissions", {}))

        if DRY_RUN:
            REPORT["actions"].append(f"would_create_{c['type']}:{c['name']}")
            continue

        try:
            if c["type"] == "text":
                await guild.create_text_channel(
                    c["name"], category=parent, overwrites=overwrites
                )
            elif c["type"] == "voice":
                await guild.create_voice_channel(
                    c["name"], category=parent, overwrites=overwrites
                )

            REPORT["actions"].append(f"create_{c['type']}:{c['name']}")
            await asyncio.sleep(1.2)
        except Exception as e:
            creation_failed = True
            REPORT["errors"].append(f"channel_create_error:{c['name']}:{e}")

    # ---- SAFETY ABORT IF CREATION FAILED ----
    if creation_failed:
        REPORT["errors"].append("ABORTED: creation failures detected, no deletions performed")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- STEP 3: DELETE FOREIGN CHANNELS ----
    allowed = {f"{c['type']}:{c['name'].lower()}" for c in spec}

    for ch in list(guild.channels):
        if ch.id in system_ids:
            continue

        if isinstance(ch, discord.CategoryChannel):
            t = "category"
        elif isinstance(ch, discord.TextChannel):
            t = "text"
        elif isinstance(ch, discord.VoiceChannel):
            t = "voice"
        else:
            continue

        if f"{t}:{ch.name.lower()}" in allowed:
            continue

        if DRY_RUN:
            REPORT["actions"].append(f"would_delete:{ch.name}")
            continue

        try:
            await ch.delete(reason="LuCha safe reconciliation")
            REPORT["actions"].append(f"delete:{ch.name}")
            await asyncio.sleep(1.2)
        except Exception as e:
            REPORT["errors"].append(f"delete_error:{ch.name}:{e}")

    # ---- FINALIZE ----
    state[today_key()] = True
    save_json(STATE_FILE, state)
    save_json(REPORT_FILE, REPORT)
    await client.close()

client.run(DISCORD_TOKEN)
