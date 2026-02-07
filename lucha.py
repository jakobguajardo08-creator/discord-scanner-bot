#!/usr/bin/env python3
"""
LuCha — SAFE Daily Channel Reconciler (JSON authority)

Fixes:
- Auto-sanitizes TEXT channel names (Discord requires lowercase + hyphens)
- Existence check is type-aware (text vs voice vs category)
- Categories created first, channels second, permissions applied, deletions last
- Hard abort if channels.json missing/empty
- Hard abort deletions if any creation fails (prevents accidental wipes)

ENV:
  DISCORD_TOKEN (required)
  GUILD_ID (required)
  LUCHA_ARMED=0/1 (default 0) -> if 0 forces DRY_RUN
  DRY_RUN=0/1 (default 0)
  FORCE_RUN=0/1 (default 0) -> bypass daily lock
  CHANNELS_FILE (default channels.json)
"""

import os
import sys
import json
import re
import asyncio
from datetime import datetime, timezone, date
from typing import Dict, Set, Tuple, Optional

import discord
from discord.utils import get

# ================= CONFIG =================
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GUILD_ID_RAW = os.getenv("GUILD_ID", "0").strip()
CHANNELS_FILE = os.getenv("CHANNELS_FILE", "channels.json")
STATE_FILE = "lucha_state.json"
REPORT_FILE = "mod_report.json"

LUCHA_ARMED = os.getenv("LUCHA_ARMED", "0").strip() == "1"
FORCE_RUN = os.getenv("FORCE_RUN", "0").strip() == "1"
DRY_RUN = os.getenv("DRY_RUN", "0").strip() == "1" or not LUCHA_ARMED

if not DISCORD_TOKEN or not GUILD_ID_RAW.isdigit():
    print("❌ Missing/invalid DISCORD_TOKEN or GUILD_ID")
    sys.exit(1)

GUILD_ID = int(GUILD_ID_RAW)

# ================= HELPERS =================
def utc_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

def load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def save_json(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def today_key() -> str:
    return f"ran:{date.today().isoformat()}"

def norm(s: str) -> str:
    return (s or "").strip().lower()

def sanitize_text_channel_name(name: str) -> str:
    """
    Discord text channel rules: lowercase, numbers, hyphens; no spaces.
    We'll convert spaces -> hyphens, remove invalid chars, collapse hyphens.
    """
    n = (name or "").strip().lower()
    n = n.replace(" ", "-")
    n = re.sub(r"[^a-z0-9\-]", "", n)
    n = re.sub(r"\-+", "-", n).strip("-")
    if not n:
        n = "channel"
    return n[:100]

def type_key(ch_type: str, name: str) -> str:
    return f"{ch_type}:{norm(name)}"

# ================= REPORT =================
REPORT = {
    "timestamp": None,
    "dry_run": DRY_RUN,
    "armed": LUCHA_ARMED,
    "actions": [],
    "errors": [],
    "notes": [],
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

def build_overwrites(guild: discord.Guild, perm_spec: dict) -> Dict[discord.abc.Snowflake, discord.PermissionOverwrite]:
    overwrites: Dict[discord.abc.Snowflake, discord.PermissionOverwrite] = {}

    everyone = guild.default_role
    if isinstance(perm_spec, dict) and "everyone" in perm_spec and isinstance(perm_spec["everyone"], dict):
        p = discord.PermissionOverwrite()
        for k, v in perm_spec["everyone"].items():
            if k in PERM_MAP:
                setattr(p, PERM_MAP[k], bool(v))
        overwrites[everyone] = p

    roles = perm_spec.get("roles", {}) if isinstance(perm_spec, dict) else {}
    if isinstance(roles, dict):
        for role_name, rules in roles.items():
            if not isinstance(rules, dict):
                continue
            role = get(guild.roles, name=str(role_name))
            if not role:
                REPORT["notes"].append(f"role_missing_skip_overwrite:{role_name}")
                continue
            p = discord.PermissionOverwrite()
            for k, v in rules.items():
                if k in PERM_MAP:
                    setattr(p, PERM_MAP[k], bool(v))
            overwrites[role] = p

    return overwrites

# ================= DISCORD CLIENT =================
intents = discord.Intents.default()
intents.guilds = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    REPORT["timestamp"] = utc_iso()

    # ---- DAILY LOCK ----
    state = load_json(STATE_FILE)
    if state.get(today_key()) and not FORCE_RUN:
        REPORT["actions"].append("exit_already_ran_today")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- LOAD & VALIDATE SPEC ----
    spec_data = load_json(CHANNELS_FILE)
    spec = spec_data.get("channels")

    if not isinstance(spec, list) or len(spec) == 0:
        REPORT["errors"].append(f"ABORTED: {CHANNELS_FILE} missing/empty/invalid")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- GET GUILD ----
    guild = client.get_guild(GUILD_ID)
    if not guild:
        REPORT["errors"].append(f"guild_not_found:{GUILD_ID}")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    me = guild.me
    if not me or not me.guild_permissions.manage_channels:
        REPORT["errors"].append("missing_manage_channels_permission")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- PROTECT SYSTEM CHANNELS ----
    system_ids = set()
    for attr in ("system_channel", "rules_channel", "public_updates_channel", "safety_alerts_channel"):
        ch = getattr(guild, attr, None)
        if ch:
            system_ids.add(ch.id)

    # ---- BUILD CURRENT TYPE-AWARE KEYS ----
    current_keys: Set[str] = set()
    for ch in guild.channels:
        if isinstance(ch, discord.CategoryChannel):
            current_keys.add(type_key("category", ch.name))
        elif isinstance(ch, discord.TextChannel):
            current_keys.add(type_key("text", ch.name))
        elif isinstance(ch, discord.VoiceChannel):
            current_keys.add(type_key("voice", ch.name))

    # ---- PREPROCESS SPEC: sanitize text channel names ----
    normalized_spec = []
    for item in spec:
        if not isinstance(item, dict):
            continue
        t = norm(str(item.get("type", "")))
        name = str(item.get("name", "")).strip()
        if not t or not name:
            continue

        # sanitize text channel names
        if t == "text":
            safe_name = sanitize_text_channel_name(name)
            if safe_name != name:
                REPORT["notes"].append(f"text_name_sanitized:{name}=>{safe_name}")
            name = safe_name

        normalized_spec.append({**item, "type": t, "name": name})

    if len(normalized_spec) == 0:
        REPORT["errors"].append("ABORTED: spec had no valid channels after normalization")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    allowed_keys = {type_key(c["type"], c["name"]) for c in normalized_spec}

    # ---- STEP 1: CREATE ALL CATEGORIES ----
    category_map: Dict[str, discord.CategoryChannel] = {}
    creation_failed = False

    for c in normalized_spec:
        if c["type"] != "category":
            continue

        k = type_key("category", c["name"])
        if k in current_keys:
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
            current_keys.add(k)
            REPORT["actions"].append(f"create_category:{c['name']}")
            await asyncio.sleep(1.2)
        except Exception as e:
            creation_failed = True
            REPORT["errors"].append(f"category_create_error:{c['name']}:{e}")

    # Also map existing categories for later
    for cat in guild.categories:
        category_map.setdefault(cat.name, cat)

    # ---- STEP 2: CREATE TEXT/VOICE CHANNELS ----
    for c in normalized_spec:
        if c["type"] == "category":
            continue

        k = type_key(c["type"], c["name"])
        if k in current_keys:
            continue

        parent = None
        cat_name = str(c.get("category", "")).strip()
        if cat_name:
            parent = category_map.get(cat_name) or get(guild.categories, name=cat_name)

        overwrites = build_overwrites(guild, c.get("permissions", {}))

        if DRY_RUN:
            REPORT["actions"].append(f"would_create_{c['type']}:{c['name']}")
            continue

        try:
            if c["type"] == "text":
                await guild.create_text_channel(c["name"], category=parent, overwrites=overwrites)
            elif c["type"] == "voice":
                await guild.create_voice_channel(c["name"], category=parent, overwrites=overwrites)
            else:
                REPORT["notes"].append(f"skip_unknown_type:{c['type']}:{c['name']}")
                continue

            current_keys.add(k)
            REPORT["actions"].append(f"create_{c['type']}:{c['name']}")
            await asyncio.sleep(1.2)
        except Exception as e:
            creation_failed = True
            REPORT["errors"].append(f"channel_create_error:{c['type']}:{c['name']}:{e}")

    # ---- SAFETY: If any creation failed, DO NOT DELETE ----
    if creation_failed:
        REPORT["errors"].append("ABORTED: creation failures occurred, deletions skipped")
        save_json(REPORT_FILE, REPORT)
        await client.close()
        return

    # ---- STEP 3: APPLY/REPAIR PERMISSIONS ON EXISTING CHANNELS ----
    # (Optional, but you asked for per-channel perms; this enforces them daily)
    for c in normalized_spec:
        perm_spec = c.get("permissions")
        if not isinstance(perm_spec, dict) or (not perm_spec.get("everyone") and not perm_spec.get("roles")):
            continue

        overwrites = build_overwrites(guild, perm_spec)

        try:
            if c["type"] == "category":
                target = get(guild.categories, name=c["name"])
            elif c["type"] == "text":
                target = get(guild.text_channels, name=c["name"])
            elif c["type"] == "voice":
                target = get(guild.voice_channels, name=c["name"])
            else:
                target = None

            if not target:
                continue

            if DRY_RUN:
                REPORT["actions"].append(f"would_set_permissions:{c['type']}:{c['name']}")
                continue

            await target.edit(overwrites=overwrites, reason="LuCha: enforce permissions from JSON")
            REPORT["actions"].append(f"set_permissions:{c['type']}:{c['name']}")
            await asyncio.sleep(1.0)

        except Exception as e:
            # permission failures shouldn't delete anything; mark as error
            REPORT["errors"].append(f"permission_edit_error:{c['type']}:{c['name']}:{e}")

    # ---- STEP 4: DELETE FOREIGN CHANNELS LAST ----
    for ch in list(guild.channels):
        if ch.id in system_ids:
            continue

        if isinstance(ch, discord.CategoryChannel):
            k = type_key("category", ch.name)
        elif isinstance(ch, discord.TextChannel):
            k = type_key("text", ch.name)
        elif isinstance(ch, discord.VoiceChannel):
            k = type_key("voice", ch.name)
        else:
            continue

        if k in allowed_keys:
            continue

        if DRY_RUN:
            REPORT["actions"].append(f"would_delete:{k}")
            continue

        try:
            await ch.delete(reason="LuCha: not in channels.json")
            REPORT["actions"].append(f"delete:{k}")
            await asyncio.sleep(1.2)
        except Exception as e:
            REPORT["errors"].append(f"delete_error:{k}:{e}")

    # ---- FINALIZE ----
    state[today_key()] = True
    save_json(STATE_FILE, state)
    save_json(REPORT_FILE, REPORT)
    await client.close()

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
