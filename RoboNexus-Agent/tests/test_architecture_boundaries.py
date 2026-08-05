import ast
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PACKAGES = ("domain", "runtime", "ports", "tools", "policies")
FORBIDDEN_PREFIXES = (
    "qwen_agent",
    "calvin",
    "backends",
    "torch",
    "transformers",
    "core.agent",
    "core.routing",
    "core.planning",
    "core.orchestration",
    "core.legacy",
)


def imported_modules(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


class ArchitectureBoundaryTest(unittest.TestCase):
    def test_removed_old_core_directories_do_not_return(self):
        removed = (
            "common",
            "localization",
            "motion",
            "perception",
            "transport",
            "verification",
            "agent",
            "execution",
            "inspection",
            "orchestration",
            "planning",
            "routing",
        )
        existing = [name for name in removed if (AGENT_ROOT / "core" / name).exists()]
        self.assertEqual(existing, [])

    def test_canonical_core_has_no_backend_model_or_legacy_imports(self):
        violations = []
        for package in CANONICAL_PACKAGES:
            for path in sorted((AGENT_ROOT / "core" / package).glob("**/*.py")):
                for module in imported_modules(path):
                    if module.startswith(FORBIDDEN_PREFIXES):
                        violations.append("{} imports {}".format(path, module))
        self.assertEqual(violations, [])

    def test_only_qwen_adapter_may_import_qwen_agent(self):
        violations = []
        for path in sorted(AGENT_ROOT.glob("**/*.py")):
            relative = path.relative_to(AGENT_ROOT)
            allowed = str(relative).startswith("adapters/qwen_agent/")
            for module in imported_modules(path):
                if module.startswith("qwen_agent") and not allowed:
                    violations.append("{} imports {}".format(relative, module))
        self.assertEqual(violations, [])

if __name__ == "__main__":
    unittest.main()
