# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                      │
# │  Project: MASX AI – Strategic Agentic AI System              │
# │  All rights reserved.                                        │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com

#!/usr/bin/env python3
"""
CLI Debug Entry Point for Flashpoint Detection

This script provides a command-line interface for testing the flashpoint detection
functionality without running the full FastAPI application. It allows direct
testing of the FlashpointLLMAgent and related services.

Usage:
    python main_flashpoint_debug.py [--iterations 5] [--target 10] [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app.agents.flashpoint_llm_agent import FlashpointLLMAgent
from app.services.token_tracker import get_token_tracker, reset_global_tracker
from app.config.logging_config import setup_logging


def main():
    """Main CLI function for flashpoint detection testing."""
    parser = argparse.ArgumentParser(
        description="Test flashpoint detection functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_flashpoint_debug.py
    python main_flashpoint_debug.py --iterations 3 --target 5
    python main_flashpoint_debug.py --verbose --save-output results.json
        """,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum search-reason cycles (default: 5)",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=10,
        help="Target number of flashpoints (default: 10)",
    )

    parser.add_argument(
        "--max-context",
        type=int,
        default=15000,
        help="Maximum context length in tokens (default: 15000)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument("--save-output", type=str, help="Save results to JSON file")

    parser.add_argument(
        "--reset-token-tracker",
        action="store_true",
        help="Reset token cost tracker before running",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)

    print("=" * 60)
    print("FLASHPOINT DETECTION DEBUG")
    print("=" * 60)
    print(f"Max iterations: {args.iterations}")
    print(f"Target flashpoints: {args.target}")
    print(f"Max context length: {args.max_context:,} tokens")
    print(f"Verbose logging: {args.verbose}")
    print("=" * 60)

    try:
        # Reset token tracker if requested
        if args.reset_token_tracker:
            reset_global_tracker()
            print("Token tracker reset")

        # Initialize agent
        print("\nInitializing FlashpointLLMAgent...")
        agent = FlashpointLLMAgent()

        # Prepare input data
        input_data = {
            "max_iterations": args.iterations,
            "target_flashpoints": args.target,
            "max_context_length": args.max_context,
            "context": {},
        }

        print(f"\nStarting flashpoint detection...")
        print(f"Configuration: {json.dumps(input_data, indent=2)}")

        # Run flashpoint detection
        result = agent.run(input_data, workflow_id="debug_run")

        if result.success:
            print("\n" + "=" * 60)
            print("FLASHPOINT DETECTION COMPLETED SUCCESSFULLY")
            print("=" * 60)

            # Display results
            data = result.data
            flashpoints = data.get("flashpoints", [])

            print(f"\nFlashpoints Found: {len(flashpoints)}")
            print(f"Iterations: {data.get('iterations', 0)}")
            print(f"Search runs: {data.get('search_runs', 0)}")
            print(f"LLM runs: {data.get('llm_runs', 0)}")

            # Display flashpoints
            if flashpoints:
                print(f"\nDetected Flashpoints:")
                print("-" * 40)
                for i, fp in enumerate(flashpoints, 1):
                    print(f"{i}. {fp['title']}")
                    print(f"   Description: {fp['description']}")
                    print(f"   Entities: {', '.join(fp['entities'])}")
                    print()

            # Display statistics
            if "geographic_distribution" in data:
                print("Geographic Distribution:")
                for country, count in data["geographic_distribution"].items():
                    print(f"  {country}: {count} flashpoints")

            # Display token usage
            token_usage = data.get("token_usage", {})
            if token_usage:
                print(f"\nToken Usage:")
                print(f"  Total calls: {token_usage.get('total_calls', 0)}")
                print(f"  Total cost: ${token_usage.get('total_cost', 0):.4f}")
                print(f"  Input tokens: {token_usage.get('total_input_tokens', 0):,}")
                print(f"  Output tokens: {token_usage.get('total_output_tokens', 0):,}")

            # Save output if requested
            if args.save_output:
                output_file = Path(args.save_output)
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\nResults saved to: {output_file}")

        else:
            print("\n" + "=" * 60)
            print("FLASHPOINT DETECTION FAILED")
            print("=" * 60)
            print(f"Error: {result.error}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    finally:
        # Print final token summary
        token_tracker = get_token_tracker()
        token_tracker.print_summary()


if __name__ == "__main__":
    main()
