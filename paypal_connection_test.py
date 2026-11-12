"""Verify PayPal Sandbox credentials without printing secrets or access tokens."""
from __future__ import annotations

import os
import sys

import requests
from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    client_id = os.getenv("PAYPAL_CLIENT_ID", "").strip()
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET", "").strip()
    base_url = os.getenv("PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com").rstrip("/")

    if not client_id or not client_secret:
        print("CONFIGURATION ERROR: PAYPAL_CLIENT_ID or PAYPAL_CLIENT_SECRET is missing.")
        print("Check that your .env file is in the same folder as this script.")
        return 1

    if "sandbox" not in base_url.lower():
        print("CONFIGURATION ERROR: This test must use the PayPal Sandbox URL.")
        print("Set PAYPAL_BASE_URL=https://api-m.sandbox.paypal.com in .env")
        return 1

    url = f"{base_url}/v1/oauth2/token"
    try:
        response = requests.post(
            url,
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
        print(f"NETWORK ERROR: Could not reach PayPal Sandbox ({type(exc).__name__}).")
        print("Check your internet connection and try again.")
        return 1

    if response.status_code != 200:
        print(f"PAYPAL AUTHENTICATION FAILED (HTTP {response.status_code}).")
        try:
            payload = response.json()
            error_name = payload.get("error")
            error_description = payload.get("error_description")
            if error_name:
                print(f"PayPal error: {error_name}")
            if error_description:
                print(f"Details: {error_description}")
        except ValueError:
            print("PayPal returned a non-JSON error response.")
        print("Recheck that both values belong to this app's Sandbox credentials.")
        return 1

    try:
        payload = response.json()
    except ValueError:
        print("UNEXPECTED RESPONSE: PayPal returned HTTP 200 but not valid JSON.")
        return 1

    if not payload.get("access_token"):
        print("UNEXPECTED RESPONSE: PayPal did not return an access token.")
        return 1

    print("SUCCESS: PayPal Sandbox credentials were accepted.")
    print(f"Token type: {payload.get('token_type', 'not provided')}")
    print(f"Token lifetime: {payload.get('expires_in', 'not provided')} seconds")
    print("The access token was not displayed or saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
