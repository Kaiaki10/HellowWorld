"""
Startup Validation

Checks all API keys, connections, and configuration before the bot starts.
Fails fast with clear error messages instead of crashing mid-pipeline.
"""

import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import aiohttp

from config import get_setting, get_env

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    required: bool = True  # False = optional (warning only)


async def check_polymarket_api() -> CheckResult:
    """Verify Polymarket API is reachable."""
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://clob.polymarket.com/markets?next_cursor=MA==&limit=1"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return CheckResult("Polymarket API", True, "Connected")
                return CheckResult("Polymarket API", False, f"Status {resp.status}")
    except Exception as e:
        return CheckResult("Polymarket API", False, f"Connection failed: {e}")


async def check_kalshi_api() -> CheckResult:
    """Verify Kalshi API is reachable."""
    base_url = get_setting("platforms", "kalshi", "base_url") or "https://demo-api.kalshi.co/trade-api/v2"
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{base_url}/markets?limit=1&status=open"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return CheckResult("Kalshi API", True, f"Connected ({base_url})")
                return CheckResult("Kalshi API", False, f"Status {resp.status}")
    except Exception as e:
        return CheckResult("Kalshi API", False, f"Connection failed: {e}")


async def check_anthropic_key() -> CheckResult:
    """Verify Anthropic API key works."""
    key = get_env("ANTHROPIC_API_KEY")
    if not key:
        return CheckResult("Anthropic API Key", False, "ANTHROPIC_API_KEY not set in .env")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "Say OK"}],
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    return CheckResult("Anthropic API Key", True, "Valid (tested with Haiku)")
                elif resp.status == 401:
                    return CheckResult("Anthropic API Key", False, "Invalid key (401 Unauthorized)")
                elif resp.status == 429:
                    return CheckResult("Anthropic API Key", True, "Valid (rate limited, but key works)")
                return CheckResult("Anthropic API Key", False, f"Status {resp.status}")
    except Exception as e:
        return CheckResult("Anthropic API Key", False, f"Connection failed: {e}")


async def check_openai_key() -> CheckResult:
    """Verify OpenAI API key works."""
    key = get_env("OPENAI_API_KEY")
    if not key:
        return CheckResult("OpenAI API Key", False, "OPENAI_API_KEY not set in .env", required=False)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    return CheckResult("OpenAI API Key", True, "Valid")
                elif resp.status == 401:
                    return CheckResult("OpenAI API Key", False, "Invalid key (401)", required=False)
                return CheckResult("OpenAI API Key", False, f"Status {resp.status}", required=False)
    except Exception as e:
        return CheckResult("OpenAI API Key", False, f"Connection failed: {e}", required=False)


async def check_google_key() -> CheckResult:
    """Verify Google AI API key works."""
    key = get_env("GOOGLE_AI_API_KEY")
    if not key:
        return CheckResult("Google AI API Key", False, "GOOGLE_AI_API_KEY not set in .env", required=False)
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    return CheckResult("Google AI API Key", True, "Valid")
                elif resp.status == 400 or resp.status == 403:
                    return CheckResult("Google AI API Key", False, "Invalid key", required=False)
                return CheckResult("Google AI API Key", False, f"Status {resp.status}", required=False)
    except Exception as e:
        return CheckResult("Google AI API Key", False, f"Connection failed: {e}", required=False)


def check_env_file() -> CheckResult:
    """Check that .env file exists."""
    env_path = Path(__file__).parent.parent / "config" / ".env"
    if env_path.exists():
        return CheckResult(".env File", True, f"Found at {env_path}")
    return CheckResult(
        ".env File", False,
        f"Not found. Copy config/.env.example to config/.env and fill in your keys"
    )


def check_polymarket_wallet() -> CheckResult:
    """Check Polymarket wallet credentials for trading."""
    key = get_env("POLYMARKET_WALLET_PRIVATE_KEY")
    api_key = get_env("POLYMARKET_API_KEY")

    if not api_key:
        paper = get_setting("general", "paper_trading", default=True)
        if paper:
            return CheckResult(
                "Polymarket Wallet", True,
                "Not configured (OK for paper trading)", required=False
            )
        return CheckResult("Polymarket Wallet", False, "POLYMARKET_API_KEY not set (required for live trading)")

    if api_key and not key:
        return CheckResult(
            "Polymarket Wallet", False,
            "API key set but POLYMARKET_WALLET_PRIVATE_KEY missing (needed for order signing)",
            required=not get_setting("general", "paper_trading", default=True),
        )

    return CheckResult("Polymarket Wallet", True, "API key and wallet configured")


def check_kalshi_credentials() -> CheckResult:
    """Check Kalshi API credentials."""
    key = get_env("KALSHI_API_KEY")
    secret = get_env("KALSHI_API_SECRET")

    if not key:
        paper = get_setting("general", "paper_trading", default=True)
        if paper:
            return CheckResult(
                "Kalshi Credentials", True,
                "Not configured (OK for paper trading)", required=False
            )
        return CheckResult("Kalshi Credentials", False, "KALSHI_API_KEY not set")

    if key and not secret:
        return CheckResult("Kalshi Credentials", False, "API key set but KALSHI_API_SECRET missing")

    return CheckResult("Kalshi Credentials", True, "API key and secret configured")


def check_data_directories() -> CheckResult:
    """Ensure data and log directories exist."""
    data_dir = Path(get_setting("general", "data_dir") or "data")
    log_dir = Path(get_setting("general", "log_dir") or "logs")

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        return CheckResult("Data Directories", True, f"{data_dir}/ and {log_dir}/ ready")
    except Exception as e:
        return CheckResult("Data Directories", False, f"Failed to create directories: {e}")


def check_config() -> CheckResult:
    """Validate critical config values."""
    issues = []

    bankroll = get_setting("general", "initial_bankroll")
    if not bankroll or bankroll <= 0:
        issues.append("initial_bankroll must be > 0")

    kelly = get_setting("risk", "kelly_fraction")
    if kelly and (kelly <= 0 or kelly > 1):
        issues.append(f"kelly_fraction {kelly} must be between 0 and 1")

    max_pos = get_setting("risk", "max_position_pct")
    if max_pos and max_pos > 20:
        issues.append(f"max_position_pct {max_pos}% seems dangerously high")

    if issues:
        return CheckResult("Configuration", False, "; ".join(issues))
    return CheckResult(
        "Configuration", True,
        f"Bankroll ${bankroll:,.0f} | Kelly {kelly} | Paper={get_setting('general', 'paper_trading', default=True)}"
    )


async def run_startup_checks(require_all: bool = False) -> bool:
    """
    Run all startup checks. Returns True if bot is safe to start.

    Args:
        require_all: If True, fail on optional checks too. Default: only fail on required.
    """
    print("=" * 60)
    print("STARTUP VALIDATION")
    print("=" * 60)

    # Sync checks
    sync_checks = [
        check_env_file(),
        check_config(),
        check_data_directories(),
        check_polymarket_wallet(),
        check_kalshi_credentials(),
    ]

    # Async checks (API connectivity)
    async_checks = await asyncio.gather(
        check_polymarket_api(),
        check_kalshi_api(),
        check_anthropic_key(),
        check_openai_key(),
        check_google_key(),
    )

    all_checks = sync_checks + list(async_checks)

    # Display results
    required_failures = []
    optional_failures = []

    for check in all_checks:
        if check.passed:
            icon = "[OK]  "
        elif check.required:
            icon = "[FAIL]"
            required_failures.append(check)
        else:
            icon = "[WARN]"
            optional_failures.append(check)

        print(f"  {icon} {check.name}: {check.detail}")

    print()

    # Count available LLM models
    llm_count = sum(1 for c in all_checks if c.name.endswith("API Key") and c.passed)
    print(f"  LLM models available: {llm_count}/3")
    if llm_count == 0:
        print("  WARNING: No LLM API keys configured. Only statistical model will be used.")
        print("  The bot will still run but prediction quality will be lower.")
    print()

    # Summary
    if required_failures:
        print(f"BLOCKED: {len(required_failures)} required check(s) failed:")
        for f in required_failures:
            print(f"  - {f.name}: {f.detail}")
        print("\nFix these before starting the bot.")
        return False

    if optional_failures:
        print(f"READY with {len(optional_failures)} warning(s):")
        for f in optional_failures:
            print(f"  - {f.name}: {f.detail}")
        print("\nBot can start. Optional features may be limited.")
    else:
        print("ALL CHECKS PASSED. Bot is ready to start.")

    print("=" * 60)
    return True
