"""Compatibility imports for CALVIN task verification."""

from backends.calvin.task_verifier import (
    CalvinOracleTaskVerifier,
    VerificationResult,
)

__all__ = ["CalvinOracleTaskVerifier", "VerificationResult"]
