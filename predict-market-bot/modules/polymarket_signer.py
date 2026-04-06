"""
Polymarket EIP-712 Order Signing

Polymarket uses off-chain order matching with on-chain settlement on Polygon.
Orders must be signed with EIP-712 typed data signatures using your wallet's
private key before they can be submitted to the CLOB API.

This module handles:
  - Building EIP-712 typed data structures for orders
  - Signing orders with eth-account
  - Generating API key headers for authenticated requests
"""

import logging
import time
import hashlib
import hmac
from typing import Optional

from eth_account import Account
from eth_account.messages import encode_structured_data

from config import get_env

logger = logging.getLogger(__name__)

# Polymarket CLOB contract addresses (Polygon mainnet)
EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
CHAIN_ID = 137  # Polygon

# EIP-712 Domain
DOMAIN = {
    "name": "Polymarket CTF Exchange",
    "version": "1",
    "chainId": CHAIN_ID,
}

# EIP-712 Order type definition
ORDER_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
    ],
    "Order": [
        {"name": "salt", "type": "uint256"},
        {"name": "maker", "type": "address"},
        {"name": "signer", "type": "address"},
        {"name": "taker", "type": "address"},
        {"name": "tokenId", "type": "uint256"},
        {"name": "makerAmount", "type": "uint256"},
        {"name": "takerAmount", "type": "uint256"},
        {"name": "expiration", "type": "uint256"},
        {"name": "nonce", "type": "uint256"},
        {"name": "feeRateBps", "type": "uint256"},
        {"name": "side", "type": "uint8"},
        {"name": "signatureType", "type": "uint8"},
    ],
}


class PolymarketSigner:
    """
    Signs Polymarket CLOB orders using EIP-712 typed data.
    """

    def __init__(self):
        self.private_key = get_env("POLYMARKET_WALLET_PRIVATE_KEY")
        self.api_key = get_env("POLYMARKET_API_KEY")
        self.api_secret = get_env("POLYMARKET_API_SECRET")
        self.account = None

        if self.private_key:
            try:
                self.account = Account.from_key(self.private_key)
                logger.info(f"Polymarket wallet loaded: {self.account.address}")
            except Exception as e:
                logger.error(f"Failed to load Polymarket wallet: {e}")

    @property
    def address(self) -> Optional[str]:
        return self.account.address if self.account else None

    @property
    def is_configured(self) -> bool:
        return self.account is not None

    def sign_order(
        self,
        token_id: str,
        side: int,             # 0 = BUY, 1 = SELL
        price: float,
        size: int,             # Number of contracts
        fee_rate_bps: int = 0,
        expiration: int = 0,   # 0 = no expiration (GTC)
        nonce: int = 0,
    ) -> Optional[dict]:
        """
        Build and sign an EIP-712 order for the Polymarket CLOB.

        Returns the signed order payload ready for API submission,
        or None if wallet is not configured.
        """
        if not self.account:
            logger.error("Cannot sign order: wallet not configured")
            return None

        # Convert price to maker/taker amounts
        # Price is 0-1, amounts are in base units (USDC has 6 decimals)
        price_raw = int(price * 1_000_000)  # USDC 6 decimals
        maker_amount = size * price_raw if side == 0 else size * 1_000_000
        taker_amount = size * 1_000_000 if side == 0 else size * price_raw

        # Generate salt (unique per order)
        salt = int(time.time() * 1000)

        if nonce == 0:
            nonce = int(time.time())

        order_data = {
            "salt": salt,
            "maker": self.account.address,
            "signer": self.account.address,
            "taker": "0x0000000000000000000000000000000000000000",
            "tokenId": int(token_id) if token_id.isdigit() else int(token_id, 16),
            "makerAmount": maker_amount,
            "takerAmount": taker_amount,
            "expiration": expiration,
            "nonce": nonce,
            "feeRateBps": fee_rate_bps,
            "side": side,
            "signatureType": 2,  # EIP-712
        }

        # Build EIP-712 structured data
        structured_data = {
            "types": ORDER_TYPES,
            "primaryType": "Order",
            "domain": DOMAIN,
            "message": order_data,
        }

        try:
            encoded = encode_structured_data(primitive=structured_data)
            signed = self.account.sign_message(encoded)

            # Build API payload
            order_payload = {
                "order": {
                    "salt": str(salt),
                    "maker": self.account.address,
                    "signer": self.account.address,
                    "taker": "0x0000000000000000000000000000000000000000",
                    "tokenId": str(order_data["tokenId"]),
                    "makerAmount": str(maker_amount),
                    "takerAmount": str(taker_amount),
                    "expiration": str(expiration),
                    "nonce": str(nonce),
                    "feeRateBps": str(fee_rate_bps),
                    "side": "BUY" if side == 0 else "SELL",
                    "signatureType": 2,
                    "signature": signed.signature.hex(),
                },
                "orderType": "GTC",
            }

            logger.debug(f"Signed order: {side} {size}@{price} for token {token_id}")
            return order_payload

        except Exception as e:
            logger.error(f"Failed to sign order: {e}")
            return None

    def generate_api_headers(self, method: str, path: str, body: str = "") -> dict:
        """
        Generate authenticated API headers for Polymarket CLOB API.
        Uses HMAC-SHA256 signing with API key/secret.
        """
        if not self.api_key or not self.api_secret:
            return {}

        timestamp = str(int(time.time()))
        message = f"{timestamp}{method}{path}{body}"
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return {
            "POLY_ADDRESS": self.address or "",
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_API_KEY": self.api_key,
        }

    def build_limit_order(
        self,
        token_id: str,
        side: str,        # "yes" or "no" → mapped to BUY/SELL
        price: float,
        size_usd: float,
    ) -> Optional[dict]:
        """
        Convenience method: build a signed limit order from human-readable inputs.

        Args:
            token_id: The condition token ID for the market
            side: "yes" (buy YES tokens) or "no" (buy NO tokens)
            price: Price per contract (0-1)
            size_usd: Total USD to spend

        Returns:
            Signed order payload ready for API, or None
        """
        contracts = int(size_usd / price) if price > 0 else 0
        if contracts <= 0:
            return None

        order_side = 0 if side == "yes" else 1  # 0=BUY, 1=SELL
        return self.sign_order(
            token_id=token_id,
            side=order_side,
            price=price,
            size=contracts,
        )
