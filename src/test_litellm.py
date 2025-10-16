#!/usr/bin/env python3
"""
Test script for LiteLLM integration in streaming_api.

Demonstrates the three generation methods:
1. generate_continuation() - Direct API call
2. generate_continuation_litellm() - LiteLLM-based call
3. generate_continuation_with_fallback() - Try direct, fallback to LiteLLM
"""

from src.streaming_api import StreamingAPIClient


def test_litellm_methods():
    """Test the different generation methods."""

    print("="*80)
    print("LiteLLM Integration Test")
    print("="*80)

    # Configure your API endpoint
    api_url = "http://localhost:5001"  # Change to your endpoint
    api_key = None  # Set if needed

    # Create client
    client = StreamingAPIClient(
        api_url=api_url,
        api_password=api_key,
        model_name="test-model"  # Change to your model
    )

    # Test context
    test_context = """The old mansion stood at the edge of town, its windows dark and empty.
    Sarah approached slowly, her footsteps echoing on the gravel path."""

    print("\n" + "="*80)
    print("Test 1: Direct API Method")
    print("="*80)
    try:
        result = client.generate_continuation(
            context=test_context,
            max_tokens=100,
            temperature=0.7
        )
        print(f"\nResult length: {len(result) if result else 0} characters")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*80)
    print("Test 2: LiteLLM Method")
    print("="*80)
    try:
        result = client.generate_continuation_litellm(
            context=test_context,
            max_tokens=100,
            temperature=0.7
        )
        print(f"\nResult length: {len(result) if result else 0} characters")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*80)
    print("Test 3: Fallback Method (tries direct, then LiteLLM)")
    print("="*80)
    try:
        result = client.generate_continuation_with_fallback(
            context=test_context,
            max_tokens=100,
            temperature=0.7
        )
        print(f"\nResult length: {len(result) if result else 0} characters")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*80)
    print("Test 4: Force LiteLLM")
    print("="*80)
    try:
        result = client.generate_continuation_with_fallback(
            context=test_context,
            max_tokens=100,
            temperature=0.7,
            use_litellm=True  # Force LiteLLM
        )
        print(f"\nResult length: {len(result) if result else 0} characters")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("\nNOTE: Make sure to configure the API URL and model name in the script!")
    print("This test requires an actual API endpoint to be running.\n")

    try:
        test_litellm_methods()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed: {e}")
        import traceback
        traceback.print_exc()
