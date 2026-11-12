"""Secure PayPal Sandbox Orders API integration for MerchantPulse AI.

The Client Secret is read only from the local .env file and is never returned
from this module. Orders are persisted locally so only orders created by this
MerchantPulse instance can be captured and duplicate captures are avoided.
"""
from __future__ import annotations

import os
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

DATABASE_PATH = ROOT_DIR / "merchantpulse_payments.sqlite3"
DEFAULT_BASE_URL = "https://api-m.sandbox.paypal.com"

# Fixed server-side catalogue: the browser never determines the charged amount.
SANDBOX_CATALOG: dict[str, tuple[str, Decimal]] = {
    "starter_demo": ("MerchantPulse Starter Demo", Decimal("5.00")),
    "growth_demo": ("MerchantPulse Growth Demo", Decimal("15.00")),
    "premium_demo": ("MerchantPulse Premium Demo", Decimal("25.00")),
}


# Refund policy enforced in code, whatever a human or an AI proposes.
REFUND_WINDOW_DAYS = 30
MAX_REFUND_REASON_LEN = 200


class PayPalIntegrationError(RuntimeError):
    """Safe, user-displayable PayPal integration error."""


def _valid_order_id(order_id: Any) -> bool:
    """PayPal order IDs are short ASCII alphanumerics; reject anything else early."""
    return (
        isinstance(order_id, str)
        and 0 < len(order_id) <= 64
        and order_id.isascii()
        and order_id.replace("-", "").isalnum()
    )


def _config() -> tuple[str, str, str]:
    client_id = os.getenv("PAYPAL_CLIENT_ID", "").strip()
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
    base_url = os.getenv("PAYPAL_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")

    if not client_id or not client_secret:
        raise PayPalIntegrationError(
            "PayPal credentials are missing. Check PAYPAL_CLIENT_ID and "
            "PAYPAL_CLIENT_SECRET in the project-root .env file."
        )
    if "sandbox" not in base_url.lower():
        raise PayPalIntegrationError(
            "Safety check blocked the request: this project is configured for "
            "Sandbox only. Set PAYPAL_BASE_URL=https://api-m.sandbox.paypal.com."
        )
    return client_id, client_secret, base_url


def credentials_configured() -> bool:
    """Return whether both credentials are present, without exposing them."""
    return bool(
        os.getenv("PAYPAL_CLIENT_ID", "").strip()
        and os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
    )



def _safe_error(response: requests.Response, action: str) -> PayPalIntegrationError:
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if not isinstance(payload, dict):
        payload = {}

    name = payload.get("name", "Unknown error")
    message = payload.get("message", "No error message returned")
    debug_id = payload.get("debug_id")
    details = payload.get("details", [])

    error_message = (
        f"PayPal {action} failed (HTTP {response.status_code}): "
        f"{name} — {message}"
    )

    if details:
        safe_details = [
            {
                "issue": item.get("issue"),
                "description": item.get("description"),
            }
            for item in details
            if isinstance(item, dict)
        ]
        error_message += f" | Details: {safe_details}"

    if debug_id:
        error_message += f" | Debug ID: {debug_id}"

    return PayPalIntegrationError(error_message)



def get_access_token() -> str:
    """Request a short-lived OAuth access token from PayPal Sandbox."""
    client_id, client_secret, base_url = _config()
    try:
        response = requests.post(
            f"{base_url}/v1/oauth2/token",
            auth=(client_id, client_secret),
            headers={
                "Accept": "application/json",
                "Accept-Language": "en_US",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise PayPalIntegrationError(
            f"Could not reach PayPal Sandbox while requesting an access token ({type(exc).__name__})."
        ) from None

    if response.status_code != 200:
        raise _safe_error(response, "authentication")
    try:
        payload = response.json()
    except ValueError:
        raise PayPalIntegrationError("PayPal returned an unreadable authentication response.") from None
    access_token = payload.get("access_token") if isinstance(payload, dict) else None
    if not access_token:
        raise PayPalIntegrationError("PayPal authentication response did not include an access token.")
    return str(access_token)

def get_paypal_order_details(order_id: str) -> dict[str, Any]:
    """Fetch an existing PayPal Sandbox order without modifying it."""
    order_id = order_id.strip() if isinstance(order_id, str) else ""
    if not order_id:
        raise PayPalIntegrationError("A PayPal order ID is required.")
    if not _valid_order_id(order_id):
        raise PayPalIntegrationError("The PayPal order ID is invalid.")

    _, _, base_url = _config()
    token = get_access_token()

    try:
        response = requests.get(
            f"{base_url}/v2/checkout/orders/{order_id}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise PayPalIntegrationError(
            f"Could not retrieve the PayPal order ({type(exc).__name__})."
        ) from None

    if response.status_code != 200:
        raise _safe_error(response, "order lookup")

    try:
        order_data = response.json()
    except ValueError:
        raise PayPalIntegrationError(
            "PayPal returned an unreadable order lookup response."
        ) from None

    if not isinstance(order_data, dict):
        raise PayPalIntegrationError("PayPal returned an unexpected order response.")

    payer = order_data.get("payer") or {}
    payment_source = (order_data.get("payment_source") or {}).get("paypal") or {}
    purchase_units = order_data.get("purchase_units") or []
    payee = (purchase_units[0].get("payee") or {}) if purchase_units else {}

    return {
    "order_id": order_data.get("id"),
    "status": order_data.get("status"),
    "payer_id": payer.get("payer_id"),
    "payer_email": payer.get("email_address"),
    "payment_source_account_id": payment_source.get("account_id"),
    "payment_source_email": payment_source.get("email_address"),
    "payee_merchant_id": payee.get("merchant_id"),
}

@contextmanager
def _db():
    """Open a SQLite connection and always close it."""
    connection = sqlite3.connect(DATABASE_PATH, timeout=45)
    connection.row_factory = sqlite3.Row

    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()



def init_database() -> None:
    """Create the local order/payment ledger if it does not exist."""
    with _db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paypal_orders (
                order_id TEXT PRIMARY KEY,
                sku TEXT NOT NULL,
                description TEXT NOT NULL,
                amount TEXT NOT NULL,
                currency TEXT NOT NULL DEFAULT 'USD',
                status TEXT NOT NULL,
                approval_url TEXT NOT NULL,
                created_at TEXT NOT NULL,
                capture_id TEXT UNIQUE,
                payer_email TEXT,
                captured_at TEXT,
                captured_amount TEXT,
                capture_currency TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_paypal_orders_status ON paypal_orders(status)"
        )
        # Audit log of refunds. One refund per order (PRIMARY KEY) means a repeat
        # click can never refund twice, even before PayPal is asked.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS paypal_refunds (
                order_id TEXT PRIMARY KEY,
                capture_id TEXT NOT NULL,
                refund_id TEXT,
                amount TEXT NOT NULL,
                currency TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                proposed_by TEXT NOT NULL,
                ai_summary TEXT,
                approved_by TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_receipts (
                receipt_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                evidence_summary TEXT NOT NULL,
                ai_reasoning TEXT NOT NULL,
                safety_status TEXT NOT NULL,
                policy_checks TEXT NOT NULL,
                human_authority TEXT NOT NULL,
                execution_status TEXT NOT NULL,
                execution_reference TEXT
            )
            """
        )


def create_decision_receipt(
    order_id: str,
    *,
    evidence_summary: str,
    ai_reasoning: str,
    safety_status: str,
    policy_checks: list[tuple[bool, str]],
    human_authority: str = "Pending explicit approval",
    execution_status: str = "Not executed",
) -> str:
    """Persist the stages of a refund decision without retaining the raw complaint."""
    init_database()
    receipt_id = str(uuid.uuid4())
    # Keep the report useful as evidence while masking email addresses and long digit runs.
    evidence = " ".join(str(evidence_summary or "").split())
    evidence = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[email masked]", evidence)
    evidence = re.sub(r"\b\d{6,}\b", "[number masked]", evidence)[:500]
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _db() as connection:
        connection.execute(
            """INSERT INTO decision_receipts
               (receipt_id, order_id, created_at, evidence_summary, ai_reasoning,
                safety_status, policy_checks, human_authority, execution_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (receipt_id, order_id, created_at, evidence, str(ai_reasoning or "")[:600],
             str(safety_status or "Not recorded")[:100], json.dumps(policy_checks),
             str(human_authority)[:100], str(execution_status)[:100]),
        )
    return receipt_id


def update_decision_receipt(receipt_id: str, **fields: str) -> None:
    """Update approval or execution status for a previously recorded receipt."""
    init_database()
    allowed = {"human_authority", "execution_status", "execution_reference"}
    values = {key: str(value)[:160] for key, value in fields.items() if key in allowed}
    if not values:
        return
    assignments = ", ".join(f"{key} = ?" for key in values)
    with _db() as connection:
        connection.execute(
            f"UPDATE decision_receipts SET {assignments} WHERE receipt_id = ?",
            (*values.values(), receipt_id),
        )


def list_decision_receipts(order_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent structured receipts, newest first."""
    init_database()
    with _db() as connection:
        if order_id:
            rows = connection.execute(
                "SELECT * FROM decision_receipts WHERE order_id = ? ORDER BY created_at DESC LIMIT ?",
                (order_id, max(1, min(int(limit), 100))),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM decision_receipts ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            ).fetchall()
    receipts = []
    for row in rows:
        item = dict(row)
        try:
            item["policy_checks"] = json.loads(item["policy_checks"])
        except (TypeError, ValueError):
            item["policy_checks"] = []
        receipts.append(item)
    return receipts


def create_order(
    sku: str,
    *,
    return_url: str,
    cancel_url: str,
) -> dict[str, Any]:
    """Create a fixed-price PayPal Sandbox order and persist its approval link."""
    if sku not in SANDBOX_CATALOG:
        raise PayPalIntegrationError("Unknown demo product. Please select one of the listed test products.")
    if not return_url.startswith(("http://localhost", "http://127.0.0.1", "https://")):
        raise PayPalIntegrationError("The PayPal return URL must be localhost or HTTPS.")
    if not cancel_url.startswith(("http://localhost", "http://127.0.0.1", "https://")):
        raise PayPalIntegrationError("The PayPal cancel URL must be localhost or HTTPS.")

    description, amount = SANDBOX_CATALOG[sku]
    token = get_access_token()
    _, _, base_url = _config()
    payload = {
        "intent": "CAPTURE",
        "purchase_units": [
            {
                "reference_id": f"merchantpulse-{sku}",
                "description": description,
                "amount": {"currency_code": "USD", "value": f"{amount:.2f}"},
            }
        ],
        "payment_source": {
            "paypal": {
                "experience_context": {
                    "brand_name": "MerchantPulse AI",
                    "shipping_preference": "NO_SHIPPING",
                    "user_action": "PAY_NOW",
                    "return_url": return_url,
                    "cancel_url": cancel_url,
                }
            }
        },
    }

    try:
        response = requests.post(
            f"{base_url}/v2/checkout/orders",
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "PayPal-Request-Id": str(uuid.uuid4()),
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        raise PayPalIntegrationError(
            f"Could not reach PayPal Sandbox while creating the order ({type(exc).__name__})."
        ) from None

    if response.status_code not in (200, 201):
        raise _safe_error(response, "order creation")
    try:
        order = response.json()
    except ValueError:
        raise PayPalIntegrationError("PayPal returned an unreadable create-order response.") from None

    order_id = order.get("id") if isinstance(order, dict) else None
    links = order.get("links", []) if isinstance(order, dict) else []
    approval_url = next(
        (
            str(link.get("href"))
            for link in links
            if link.get("rel") in {"approve", "payer-action"} and link.get("href")
        ),
        None,
    )
    if not order_id or not approval_url:
        raise PayPalIntegrationError(
            "PayPal created a response without the expected order ID or buyer-approval link. "
            "Check the Sandbox app and try again."
        )
    
    init_database()
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _db() as connection:
        connection.execute(
            """
            INSERT INTO paypal_orders
                (order_id, sku, description, amount, currency, status, approval_url, created_at)
            VALUES (?, ?, ?, ?, 'USD', ?, ?, ?)
            """,
            (
                str(order_id), sku, description, f"{amount:.2f}",
                str(order.get("status", "CREATED")), approval_url, created_at,
            ),
        )
    return {
        "order_id": str(order_id),
        "sku": sku,
        "description": description,
        "amount": f"{amount:.2f}",
        "currency": "USD",
        "status": str(order.get("status", "CREATED")),
        "approval_url": approval_url,
    }


def get_order(order_id: str) -> dict[str, Any] | None:
    init_database()
    with _db() as connection:
        row = connection.execute(
            "SELECT * FROM paypal_orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    return dict(row) if row else None


def _capture_resource(order_data: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first well-formed capture resource in a PayPal order."""
    purchase_units = order_data.get("purchase_units")
    if not isinstance(purchase_units, list):
        return None

    for unit in purchase_units:
        if not isinstance(unit, dict):
            continue
        payments = unit.get("payments")
        if not isinstance(payments, dict):
            continue
        captures = payments.get("captures")
        if not isinstance(captures, list):
            continue
        for capture in captures:
            if isinstance(capture, dict):
                return capture
    return None


def _already_captured_result(order_id: str, local_order: dict[str, Any]) -> dict[str, Any]:
    """Build a consistent result for a capture already saved in SQLite."""
    return {
        "order_id": order_id,
        "capture_id": local_order["capture_id"],
        "status": "COMPLETED",
        "amount": local_order.get("captured_amount") or local_order["amount"],
        "currency": local_order.get("capture_currency") or local_order["currency"],
        "payer_email": local_order.get("payer_email"),
        "captured_at": local_order.get("captured_at"),
        "already_captured": True,
    }


def capture_order(order_id: str) -> dict[str, Any]:
    """Capture one locally-created order, serializing duplicate requests safely.

    BEGIN IMMEDIATE holds a SQLite write reservation while PayPal is contacted.
    A concurrent callback waits, then re-reads the row and returns the saved
    capture instead of sending a second capture request. If this process exits,
    SQLite releases the transaction automatically; no stale in-progress flag
    or database migration is needed.
    """
    if not _valid_order_id(order_id):
        raise PayPalIntegrationError("The returned PayPal order ID is invalid.")

    try:
        local_order = get_order(order_id)
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            raise PayPalIntegrationError(
                "Another payment capture is still being processed. Refresh the dashboard and "
                "check the Sandbox order status before trying again."
            ) from None
        raise

    if not local_order:
        raise PayPalIntegrationError(
            "This order was not created by this MerchantPulse app. Start a new Sandbox checkout."
        )
    if local_order["status"] == "COMPLETED" and local_order.get("capture_id"):
        return _already_captured_result(order_id, local_order)

    token = get_access_token()
    _, _, base_url = _config()
    capture_url = f"{base_url}/v2/checkout/orders/{order_id}/capture"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
        # Reusing the same request ID makes a retry idempotent at PayPal.
        "PayPal-Request-Id": f"capture-{order_id}",
    }

    try:
        with _db() as connection:
            # Acquire the write lock before re-checking status. A second callback
            # cannot pass this point until the first callback commits or rolls back.
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM paypal_orders WHERE order_id = ?", (order_id,)
            ).fetchone()
            if row is None:
                raise PayPalIntegrationError(
                    "This order was not created by this MerchantPulse app. Start a new Sandbox checkout."
                )

            locked_order = dict(row)
            if locked_order["status"] == "COMPLETED" and locked_order.get("capture_id"):
                return _already_captured_result(order_id, locked_order)

            try:
                response = requests.post(
                    capture_url,
                    json={},
                    headers=headers,
                    timeout=30,
                )
            except requests.RequestException as capture_exc:
                # A timeout is ambiguous: PayPal may have completed the capture.
                # Look up the order before telling the caller what happened.
                try:
                    lookup = requests.get(
                        f"{base_url}/v2/checkout/orders/{order_id}",
                        headers={
                            "Authorization": f"Bearer {token}",
                            "Content-Type": "application/json",
                        },
                        timeout=10,
                    )
                    lookup_data = lookup.json() if lookup.status_code == 200 else {}
                except (requests.RequestException, ValueError):
                    lookup_data = {}

                if not isinstance(lookup_data, dict) or lookup_data.get("status") != "COMPLETED":
                    raise PayPalIntegrationError(
                        "PayPal did not return a capture confirmation and the order could not be "
                        "verified as completed. Check this order in PayPal Sandbox before retrying."
                    ) from None
                order_data = lookup_data
            else:
                if response.status_code in (200, 201):
                    try:
                        order_data = response.json()
                    except ValueError:
                        raise PayPalIntegrationError(
                            "PayPal returned an unreadable capture response. Check the order in "
                            "Sandbox before retrying."
                        ) from None
                else:
                    # Recover if PayPal completed the capture but the response was an error
                    # or an earlier run stopped before the local SQLite update.
                    try:
                        lookup = requests.get(
                            f"{base_url}/v2/checkout/orders/{order_id}",
                            headers={
                                "Authorization": f"Bearer {token}",
                                "Content-Type": "application/json",
                            },
                            timeout=10,
                        )
                        lookup_data = lookup.json() if lookup.status_code == 200 else {}
                    except (requests.RequestException, ValueError):
                        lookup_data = {}

                    if not isinstance(lookup_data, dict) or lookup_data.get("status") != "COMPLETED":
                        raise _safe_error(response, "payment capture")
                    order_data = lookup_data

            if not isinstance(order_data, dict):
                raise PayPalIntegrationError("PayPal returned an unexpected capture response.")

            capture = _capture_resource(order_data)
            if order_data.get("status") != "COMPLETED" or not capture or capture.get("status") != "COMPLETED":
                raise PayPalIntegrationError(
                    "PayPal has not confirmed a completed capture. Check the order in Sandbox before retrying."
                )

            capture_id = capture.get("id")
            capture_amount = capture.get("amount")
            if not isinstance(capture_amount, dict):
                capture_amount = {}
            captured_amount = capture_amount.get("value")
            captured_currency = capture_amount.get("currency_code")
            try:
                expected_amount = Decimal(locked_order["amount"])
                actual_amount = Decimal(str(captured_amount))
            except (InvalidOperation, TypeError, ValueError):
                raise PayPalIntegrationError(
                    "PayPal's capture response did not contain a valid amount."
                ) from None

            if actual_amount != expected_amount or captured_currency != locked_order["currency"]:
                raise PayPalIntegrationError(
                    "The captured amount or currency did not match the original MerchantPulse order. "
                    "The payment was not added to the dashboard ledger."
                )
            if not capture_id:
                raise PayPalIntegrationError(
                    "PayPal confirmed the capture but did not return a capture ID."
                )

            payer = order_data.get("payer")
            payer_email = payer.get("email_address") if isinstance(payer, dict) else None
            captured_at = (
                capture.get("update_time")
                or capture.get("create_time")
                or datetime.now(timezone.utc).isoformat()
            )
            update = connection.execute(
                """
                UPDATE paypal_orders
                SET status = 'COMPLETED',
                    capture_id = ?,
                    payer_email = ?,
                    captured_at = ?,
                    captured_amount = ?,
                    capture_currency = ?
                WHERE order_id = ? AND status != 'COMPLETED'
                """,
                (
                    str(capture_id),
                    payer_email,
                    str(captured_at),
                    str(actual_amount),
                    str(captured_currency),
                    order_id,
                ),
            )
            if update.rowcount != 1:
                raise PayPalIntegrationError(
                    "The local payment record could not be updated. Check the order in Sandbox "
                    "before retrying."
                )

            return {
                "order_id": order_id,
                "capture_id": str(capture_id),
                "status": "COMPLETED",
                "amount": str(actual_amount),
                "currency": str(captured_currency),
                "payer_email": payer_email,
                "captured_at": str(captured_at),
                "already_captured": False,
            }
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            raise PayPalIntegrationError(
                "Another payment capture is still being processed. Refresh the dashboard and "
                "check the Sandbox order status before trying again."
            ) from None
        raise


def list_completed_payments() -> list[dict[str, Any]]:
    """Return verified completed Sandbox captures (with refund status) for the dashboard."""
    init_database()
    with _db() as connection:
        rows = connection.execute(
            """
            SELECT o.order_id, o.capture_id, o.description, o.captured_amount AS amount,
                   o.capture_currency AS currency, o.status, o.payer_email, o.captured_at,
                   r.status AS refund_status, r.refund_id AS refund_id,
                   r.created_at AS refunded_at
            FROM paypal_orders o
            LEFT JOIN paypal_refunds r ON r.order_id = o.order_id
            WHERE o.status = 'COMPLETED' AND o.capture_id IS NOT NULL
            ORDER BY o.captured_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Refunds (Sandbox only). Money-moving, so every guard below is enforced here,
# in code, regardless of who or what proposed the refund.
# ---------------------------------------------------------------------------


def get_refund(order_id: str) -> dict[str, Any] | None:
    init_database()
    with _db() as connection:
        row = connection.execute(
            "SELECT * FROM paypal_refunds WHERE order_id = ?", (order_id,)
        ).fetchone()
    return dict(row) if row else None


def _refund_result(row: dict[str, Any], *, already: bool) -> dict[str, Any]:
    return {
        "order_id": row["order_id"],
        "capture_id": row["capture_id"],
        "refund_id": row.get("refund_id"),
        "status": row["status"],
        "amount": row["amount"],
        "currency": row["currency"],
        "reason": row["reason"],
        "already_refunded": already,
    }


def _parse_paypal_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def check_refund_policy(local_order: dict[str, Any] | None) -> list[tuple[bool, str]]:
    """Return (passed, message) policy checks for a locally stored order."""
    if not local_order:
        return [(False, "Order was not created by this MerchantPulse app.")]
    checks: list[tuple[bool, str]] = []
    captured = local_order.get("status") == "COMPLETED" and bool(local_order.get("capture_id"))
    checks.append((captured, "Payment was captured and verified." if captured else "Only completed captures can be refunded."))
    when = _parse_paypal_time(local_order.get("captured_at"))
    if when is None:
        checks.append((False, "Capture time could not be verified, so the refund window cannot be checked."))
    else:
        age_days = (datetime.now(timezone.utc) - when).total_seconds() / 86400
        ok = age_days <= REFUND_WINDOW_DAYS
        checks.append(
            (ok, f"Within the {REFUND_WINDOW_DAYS}-day refund window ({age_days:.1f} days old)."
             if ok else f"Outside the {REFUND_WINDOW_DAYS}-day refund window ({age_days:.0f} days old).")
        )
    existing = get_refund(str(local_order.get("order_id", "")))
    already = bool(existing and existing["status"] in {"COMPLETED", "PENDING"})
    checks.append((not already, "Not refunded yet." if not already else "This payment was already refunded."))
    return checks


def _lookup_refunded_state(base_url: str, token: str, order_id: str) -> dict[str, Any] | None:
    """Ask PayPal whether this order's capture is already refunded (None if unknown/no)."""
    try:
        lookup = requests.get(
            f"{base_url}/v2/checkout/orders/{order_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=10,
        )
        data = lookup.json() if lookup.status_code == 200 else {}
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    capture = _capture_resource(data)
    if not capture or str(capture.get("status", "")).upper() != "REFUNDED":
        return None
    refund_id = None
    for unit in data.get("purchase_units") or []:
        refunds = ((unit or {}).get("payments") or {}).get("refunds") or []
        if refunds and isinstance(refunds[0], dict):
            refund_id = refunds[0].get("id")
            break
    return {"refund_id": refund_id, "status": "COMPLETED"}


def refund_capture(
    order_id: str,
    *,
    reason: str,
    approved_by: str = "merchant",
    proposed_by: str = "manual",
    ai_summary: str | None = None,
) -> dict[str, Any]:
    """Fully refund one captured Sandbox payment after explicit human approval.

    Safeguards: order must be ours and captured; inside the refund window; one
    refund per order (database primary key); writes are serialised with
    BEGIN IMMEDIATE; the PayPal-Request-Id is deterministic so a retry after a
    timeout can never refund twice; if PayPal's answer is ambiguous we look the
    order up before deciding.
    """
    if not _valid_order_id(order_id):
        raise PayPalIntegrationError("The PayPal order ID is invalid.")
    reason = " ".join(str(reason or "").split())[:MAX_REFUND_REASON_LEN]
    if not reason:
        raise PayPalIntegrationError("A refund reason is required.")
    if proposed_by not in {"manual", "ai"}:
        raise PayPalIntegrationError("Unknown refund source.")

    local_order = get_order(order_id)
    if not local_order:
        raise PayPalIntegrationError(
            "This order was not created by this MerchantPulse app, so it cannot be refunded."
        )
    existing = get_refund(order_id)
    if existing and existing["status"] in {"COMPLETED", "PENDING"}:
        return _refund_result(existing, already=True)
    for passed, message in check_refund_policy(local_order):
        if not passed:
            raise PayPalIntegrationError(f"Refund blocked by policy: {message}")

    token = get_access_token()
    _, _, base_url = _config()
    capture_id = str(local_order["capture_id"])
    amount = str(local_order.get("captured_amount") or local_order["amount"])
    currency = str(local_order.get("capture_currency") or local_order["currency"])

    try:
        with _db() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM paypal_refunds WHERE order_id = ?", (order_id,)
            ).fetchone()
            if row is not None and dict(row)["status"] in {"COMPLETED", "PENDING"}:
                return _refund_result(dict(row), already=True)

            try:
                response = requests.post(
                    f"{base_url}/v2/payments/captures/{capture_id}/refund",
                    json={
                        "amount": {"value": amount, "currency_code": currency},
                        "note_to_payer": reason,
                    },
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "Prefer": "return=representation",
                        # Deterministic: a retry returns the same refund, never a second one.
                        "PayPal-Request-Id": f"refund-{order_id}",
                    },
                    timeout=30,
                )
            except requests.RequestException:
                state = _lookup_refunded_state(base_url, token, order_id)
                if state is None:
                    raise PayPalIntegrationError(
                        "PayPal did not confirm the refund and it could not be verified. "
                        "Retrying is safe: the same request ID is reused, so it cannot refund twice."
                    ) from None
                refund_id, status = state["refund_id"], state["status"]
            else:
                if response.status_code in (200, 201):
                    try:
                        payload = response.json()
                    except ValueError:
                        raise PayPalIntegrationError(
                            "PayPal returned an unreadable refund response. Check the order in Sandbox before retrying."
                        ) from None
                    if not isinstance(payload, dict):
                        raise PayPalIntegrationError("PayPal returned an unexpected refund response.")
                    refund_id, status = payload.get("id"), str(payload.get("status", "")).upper()
                else:
                    # e.g. CAPTURE_FULLY_REFUNDED: it may already be refunded at PayPal.
                    state = _lookup_refunded_state(base_url, token, order_id)
                    if state is None:
                        raise _safe_error(response, "refund")
                    refund_id, status = state["refund_id"], state["status"]

            if status not in {"COMPLETED", "PENDING"}:
                raise PayPalIntegrationError(
                    f"PayPal did not complete the refund (status: {status or 'unknown'}). Nothing was recorded."
                )

            created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            connection.execute(
                """
                INSERT OR REPLACE INTO paypal_refunds
                    (order_id, capture_id, refund_id, amount, currency, status, reason,
                     proposed_by, ai_summary, approved_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_id, capture_id, str(refund_id) if refund_id else None, amount, currency,
                    status, reason, proposed_by, (ai_summary or "")[:600] or None,
                    " ".join(str(approved_by).split())[:60] or "merchant", created_at,
                ),
            )
            return {
                "order_id": order_id, "capture_id": capture_id,
                "refund_id": str(refund_id) if refund_id else None, "status": status,
                "amount": amount, "currency": currency, "reason": reason,
                "already_refunded": False,
            }
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            raise PayPalIntegrationError(
                "Another refund is still being processed. Refresh the dashboard before trying again."
            ) from None
        raise
