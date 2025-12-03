import discord
import datetime
from ai_client import ai_generate

async def generate_daily_summary(channel: discord.TextChannel):
    messages = []
    async for m in channel.history(limit=300):
        if not m.author.bot:
            messages.append(f"{m.author.display_name}: {m.content}")

    history_text = "\n".join(messages[-200:])

    prompt = f"""
    Summarize the following Discord chat history in a fun way.
    Include: gossip-style recap, top chaos moments, and a silly headline.
    Chat history:
    {history_text}
    """

    summary = await ai_generate(prompt)
    return summary


async def daily_fortune():
    prompt = """
    Generate three fortune cookie messages:
    1) wholesome
    2) chaotic
    3) absurdly specific
    Format with bullet points.
    """
    return await ai_generate(prompt)


async def daily_server_event():
    prompt = """
    Generate one fun server-wide event for today. 
    Examples: "emoji-only day", "pirate speech day", "no using the letter E", etc.
    Make it creative.
    """
    return await ai_generate(prompt)
