"""Compatibility imports for CALVIN task verification."""

from backends.calvin.verification.task_verifier import (
    CalvinOracleTaskVerifier,
    VerificationResult,
)

__all__ = ["CalvinOracleTaskVerifier", "VerificationResult"]
