import unittest

from backends.calvin.runtime.tool_session_server import CalvinToolSessionServer
from core.domain import AgentSessionStatus


class _Session:
    def __init__(self, status):
        self.status = status


class ToolSessionLifecycleTest(unittest.TestCase):
    def test_terminal_callback_runs_exactly_once(self):
        calls = []
        server = object.__new__(CalvinToolSessionServer)
        server.session = _Session(AgentSessionStatus.SUCCEEDED)
        server.on_terminal = lambda: calls.append("closed")
        server._terminal_notified = False

        server._notify_terminal()
        server._notify_terminal()

        self.assertEqual(calls, ["closed"])

    def test_active_session_does_not_release_clients(self):
        calls = []
        server = object.__new__(CalvinToolSessionServer)
        server.session = _Session(AgentSessionStatus.ACTIVE)
        server.on_terminal = lambda: calls.append("closed")
        server._terminal_notified = False

        server._notify_terminal()

        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
