import argparse
import json

from env.baseline import run_baseline


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline inference across all OpenEnv tasks.",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Run deterministic fallback policy even if API keys are set.",
    )
    args = parser.parse_args()

    result = run_baseline(force_policy="fallback" if args.force_fallback else None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
