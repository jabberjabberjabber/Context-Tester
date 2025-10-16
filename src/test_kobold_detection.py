#!/usr/bin/env python3
"""
Test script to demonstrate KoboldCpp API detection.
"""

from src.streaming_api import StreamingAPIClient


def test_kobold_detection():
    """Test the is_kobold_api() method with different API endpoints."""

    # Test with a local KoboldCpp instance (if running)
    print("Testing KoboldCpp detection...\n")

    test_urls = [
        "http://localhost:5001",
        "http://172.16.0.219:5001",
        "https://integrate.api.nvidia.com",
    ]

    for url in test_urls:
        print(f"Testing: {url}")
        try:
            client = StreamingAPIClient(api_url=url)
            is_kobold = client.is_kobold_api()

            if is_kobold:
                print(f"  ✓ Detected as KoboldCpp API")
            else:
                print(f"  ✗ Not a KoboldCpp API")

        except Exception as e:
            print(f"  ! Error: {e}")

        print()


if __name__ == "__main__":
    test_kobold_detection()
