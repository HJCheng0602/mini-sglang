"""Run all PD tests."""
from __future__ import annotations

import subprocess
import sys
import traceback
from typing import Callable, List, Tuple

from minisgl.utils import init_logger

logger = init_logger(__name__)

# Tests that load models must run in separate processes (Engine can only init once)
MODEL_TESTS = [
    "tests.pd.test_prefill_scheduler",
    "tests.pd.test_prefill_worker",
    "tests.pd.test_decode_scheduler",
    "tests.pd.test_decode_worker",
    "tests.pd.test_integration",
]


def run_test(test_func: Callable, test_name: str) -> bool:
    """Run a single test function."""
    try:
        test_func()
        return True
    except Exception as e:
        logger.error(f"FAILED: {test_name}")
        logger.error(f"  Error: {e}")
        traceback.print_exc()
        return False


def run_model_test(test_module: str) -> bool:
    """Run a model test in a subprocess."""
    logger.info(f"\nRunning: {test_module}")
    result = subprocess.run(
        [sys.executable, "-m", test_module],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode == 0:
        # Extract last few lines for summary
        lines = result.stdout.strip().split("\n")
        for line in lines[-5:]:
            logger.info(f"  {line}")
        return True
    else:
        logger.error(f"FAILED: {test_module}")
        logger.error(result.stdout[-500:] if result.stdout else "")
        logger.error(result.stderr[-500:] if result.stderr else "")
        return False


def collect_basic_tests() -> List[Tuple[Callable, str]]:
    """Collect tests that don't require model loading."""
    tests = []

    # test_base.py
    from tests.pd.test_base import (
        test_transfer_status_enum,
        test_kv_transfer_args_basic,
        test_kv_transfer_args_with_ptrs,
        test_kv_transfer_args_edge_cases,
        test_base_kv_transfer_backend_interface,
    )
    tests.extend([
        (test_transfer_status_enum, "test_base.test_transfer_status_enum"),
        (test_kv_transfer_args_basic, "test_base.test_kv_transfer_args_basic"),
        (test_kv_transfer_args_with_ptrs, "test_base.test_kv_transfer_args_with_ptrs"),
        (test_kv_transfer_args_edge_cases, "test_base.test_kv_transfer_args_edge_cases"),
        (test_base_kv_transfer_backend_interface, "test_base.test_base_kv_transfer_backend_interface"),
    ])

    # test_message.py
    from tests.pd.test_message import (
        test_prefill_done_msg,
        test_kv_transfer_req,
        test_kv_transfer_ack,
        test_kv_transfer_ack_with_error,
        test_prefill_worker_ready,
        test_decode_worker_ready,
        test_message_roundtrip,
    )
    tests.extend([
        (test_prefill_done_msg, "test_message.test_prefill_done_msg"),
        (test_kv_transfer_req, "test_message.test_kv_transfer_req"),
        (test_kv_transfer_ack, "test_message.test_kv_transfer_ack"),
        (test_kv_transfer_ack_with_error, "test_message.test_kv_transfer_ack_with_error"),
        (test_prefill_worker_ready, "test_message.test_prefill_worker_ready"),
        (test_decode_worker_ready, "test_message.test_decode_worker_ready"),
        (test_message_roundtrip, "test_message.test_message_roundtrip"),
    ])

    # test_transfer.py
    from tests.pd.test_transfer import (
        test_backend_registry,
        test_create_gloo_backend,
        test_create_nccl_backend,
        test_create_invalid_backend,
        test_gloo_backend_init_transfer,
        test_gloo_backend_poll_nonexistent,
        test_gloo_backend_cleanup_nonexistent,
        test_kv_transfer_args_creation,
        test_transfer_status_ordering,
        test_multiple_backends,
    )
    tests.extend([
        (test_backend_registry, "test_transfer.test_backend_registry"),
        (test_create_gloo_backend, "test_transfer.test_create_gloo_backend"),
        (test_create_nccl_backend, "test_transfer.test_create_nccl_backend"),
        (test_create_invalid_backend, "test_transfer.test_create_invalid_backend"),
        (test_gloo_backend_init_transfer, "test_transfer.test_gloo_backend_init_transfer"),
        (test_gloo_backend_poll_nonexistent, "test_transfer.test_gloo_backend_poll_nonexistent"),
        (test_gloo_backend_cleanup_nonexistent, "test_transfer.test_gloo_backend_cleanup_nonexistent"),
        (test_kv_transfer_args_creation, "test_transfer.test_kv_transfer_args_creation"),
        (test_transfer_status_ordering, "test_transfer.test_transfer_status_ordering"),
        (test_multiple_backends, "test_transfer.test_multiple_backends"),
    ])

    return tests


def run_all_tests() -> None:
    """Run all PD tests."""
    logger.info("=" * 60)
    logger.info("Running PD Tests")
    logger.info("=" * 60)

    passed = 0
    failed = 0

    # 1. Basic tests (in-process)
    logger.info("\n--- Basic Tests (no model) ---")
    for test_func, test_name in collect_basic_tests():
        logger.info(f"\nRunning: {test_name}")
        if run_test(test_func, test_name):
            passed += 1
            logger.info("  PASSED")
        else:
            failed += 1

    # 2. Model tests (subprocess)
    logger.info("\n--- Model Tests (subprocess) ---")
    for test_module in MODEL_TESTS:
        if run_model_test(test_module):
            passed += 1
            logger.info("  PASSED")
        else:
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed} passed, {failed} failed, {passed + failed} total")
    logger.info("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
