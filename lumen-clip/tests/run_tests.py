#!/usr/bin/env python3
"""
Test runner script for Lumen-CLIP with configuration-based testing.

This script provides different test modes:
- Unit tests: Test individual components in isolation
- Integration tests: Test component interactions
- Preprocessing tests: Test image preprocessing and normalization
- Model tests: Test model loading and inference
- All tests: Run complete test suite

Usage:
    python tests/run_tests.py [--unit|--integration|--preprocessing|--model|--all] [--torch|--onnx|--both]
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def run_pytest(args, extra_args=None):
    """Run pytest with given arguments."""
    cmd = ["python", "-m", "pytest"] + args

    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_unit_tests(runtime=None):
    """Run unit tests."""
    args = ["tests/unit", "-v", "-m", "unit"]

    if runtime:
        if runtime == "torch":
            args.extend(["-k", "torch or not onnx"])
        elif runtime == "onnx":
            args.extend(["-k", "onnx or not torch"])

    return run_pytest(args)


def run_integration_tests(runtime=None):
    """Run integration tests."""
    args = ["tests/integration", "-v", "-m", "integration"]

    if runtime:
        args.extend(["-k", runtime])

    return run_pytest(args)


def run_preprocessing_tests():
    """Run preprocessing pipeline tests."""
    args = [
        "tests/unit/test_preprocessing/",
        "-v",
        "-m", "preprocessing or normalization or precision"
    ]

    return run_pytest(args)


def run_model_tests(runtime=None):
    """Run model loading and inference tests."""
    args = ["tests/unit/test_models/", "-v", "-m", "model_loading"]

    if runtime:
        if runtime == "torch":
            args.extend(["-k", "torch"])
        elif runtime == "onnx":
            args.extend(["-k", "onnx"])

    return run_pytest(args)


def run_backend_tests(runtime=None):
    """Run backend-specific tests."""
    args = ["tests/unit/test_backends/", "-v"]

    if runtime == "torch":
        args.extend(["-k", "torch"])
    elif runtime == "onnx":
        args.extend(["-k", "onnx"])

    return run_pytest(args)


def run_all_tests(runtime=None):
    """Run all tests."""
    args = ["tests/", "-v"]

    if runtime:
        if runtime == "torch":
            args.extend(["-k", "torch or not onnx"])
        elif runtime == "onnx":
            args.extend(["-k", "onnx or not torch"])

    return run_pytest(args)


def run_quick_tests():
    """Run a quick subset of tests for development."""
    args = [
        "tests/unit/test_preprocessing/test_normalization.py",
        "tests/unit/test_models/test_clip_model.py::TestCLIPModelManager::test_encode_image",
        "tests/unit/test_models/test_clip_model.py::TestEmbeddingQuality",
        "-v"
    ]

    return run_pytest(args)


def run_debug_tests():
    """Run tests with debug output."""
    args = [
        "tests/unit/test_preprocessing/test_normalization.py",
        "-v",
        "-s",  # Don't capture output
        "--tb=long",  # Long traceback format
        "-p", "no:warnings"  # Disable warnings
    ]

    return run_pytest(args)


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Lumen-CLIP Test Runner")

    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true", help="Run unit tests")
    test_group.add_argument("--integration", action="store_true", help="Run integration tests")
    test_group.add_argument("--preprocessing", action="store_true", help="Run preprocessing tests")
    test_group.add_argument("--model", action="store_true", help="Run model tests")
    test_group.add_argument("--backend", action="store_true", help="Run backend tests")
    test_group.add_argument("--all", action="store_true", help="Run all tests")
    test_group.add_argument("--quick", action="store_true", help="Run quick tests")
    test_group.add_argument("--debug", action="store_true", help="Run tests with debug output")

    # Runtime selection
    runtime_group = parser.add_mutually_exclusive_group()
    runtime_group.add_argument("--torch", action="store_true", help="Test Torch runtime only")
    runtime_group.add_argument("--onnx", action="store_true", help="Test ONNX runtime only")
    runtime_group.add_argument("--both", action="store_true", help="Test both runtimes")

    # Additional options
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")

    args = parser.parse_args()

    # Determine runtime
    runtime = None
    if args.torch:
        runtime = "torch"
    elif args.onnx:
        runtime = "onnx"
    elif args.both:
        runtime = None  # Test both

    # Build extra arguments
    extra_args = []
    if args.coverage:
        extra_args.extend(["--cov=lumen_clip", "--cov-report=html", "--cov-report=term"])
    if args.verbose:
        extra_args.append("-vv")
    if args.slow:
        extra_args.append("--run-slow")

    # Run selected tests
    try:
        if args.unit:
            return run_unit_tests(runtime)
        elif args.integration:
            return run_integration_tests(runtime)
        elif args.preprocessing:
            return run_preprocessing_tests()
        elif args.model:
            return run_model_tests(runtime)
        elif args.backend:
            return run_backend_tests(runtime)
        elif args.all:
            return run_all_tests(runtime)
        elif args.quick:
            return run_quick_tests()
        elif args.debug:
            return run_debug_tests()
        else:
            # Default: run unit tests
            print("Running unit tests (default)")
            return run_unit_tests(runtime)

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())