"""Tests for ContainerIO — the file I/O bridge between Aider and ARES containers.

Tests that read_text() and write_text() correctly route through
container.exec_run() instead of the local filesystem.
"""

import asyncio
import base64

import pytest

from ares.code_agents.aider_agent import container_io
from ares.containers import containers
from ares.testing.mock_container import MockContainer


def _make_container_io(container: MockContainer) -> container_io.ContainerIO:
    """Helper to create a ContainerIO with a real event loop."""
    loop = asyncio.get_running_loop()
    return container_io.ContainerIO(container=container, event_loop=loop)


class TestContainerIOReadText:
    """Test read_text — reading files from the container."""

    @pytest.mark.asyncio
    async def test_read_text_success(self):
        """Test reading a file returns its content."""

        def handler(command: str) -> containers.ExecResult:
            if "cat" in command and "main.py" in command:
                return containers.ExecResult(output="print('hello')\n", exit_code=0)
            return containers.ExecResult(output="", exit_code=0)

        container = MockContainer(exec_handler=handler)

        cio = _make_container_io(container)
        result = await asyncio.to_thread(cio.read_text, "/workspace/main.py")

        assert result == "print('hello')\n"
        assert any("cat" in cmd and "main.py" in cmd for cmd in container.exec_commands)

    @pytest.mark.asyncio
    async def test_read_text_file_not_found(self):
        """Test reading a nonexistent file returns None."""

        def handler(command: str) -> containers.ExecResult:
            if "cat" in command and "missing.py" in command:
                return containers.ExecResult(output="cat: no such file", exit_code=1)
            return containers.ExecResult(output="", exit_code=0)

        container = MockContainer(exec_handler=handler)

        cio = _make_container_io(container)
        result = await asyncio.to_thread(cio.read_text, "/workspace/missing.py")

        assert result is None

    @pytest.mark.asyncio
    async def test_read_text_silent_mode(self):
        """Test read_text with silent=True suppresses error logging."""

        def handler(command: str) -> containers.ExecResult:
            if "cat" in command:
                return containers.ExecResult(output="cat: no such file", exit_code=1)
            return containers.ExecResult(output="", exit_code=0)

        container = MockContainer(exec_handler=handler)

        cio = _make_container_io(container)
        # Should not raise, just return None
        result = await asyncio.to_thread(cio.read_text, "/workspace/missing.py", silent=True)

        assert result is None

    @pytest.mark.asyncio
    async def test_read_text_shell_escapes_filename(self):
        """Test filenames with special characters are properly escaped."""

        def handler(command: str) -> containers.ExecResult:
            if "cat" in command and "file with spaces" in command:
                return containers.ExecResult(output="content", exit_code=0)
            return containers.ExecResult(output="", exit_code=0)

        container = MockContainer(exec_handler=handler)

        cio = _make_container_io(container)
        result = await asyncio.to_thread(cio.read_text, "/workspace/file with spaces.py")

        assert result == "content"

    @pytest.mark.asyncio
    async def test_read_text_exception_returns_none(self):
        """Test read_text returns None on container exception."""

        def raise_on_exec(command: str) -> containers.ExecResult:  # noqa: ARG001
            raise RuntimeError("Container connection lost")

        container = MockContainer(exec_handler=raise_on_exec)

        cio = _make_container_io(container)
        result = await asyncio.to_thread(cio.read_text, "/workspace/file.py")

        assert result is None


class TestContainerIOWriteText:
    """Test write_text — writing files into the container."""

    @pytest.mark.asyncio
    async def test_write_text_creates_parent_dir(self):
        """Test write_text creates parent directories before writing."""
        container = MockContainer()

        cio = _make_container_io(container)
        await asyncio.to_thread(cio.write_text, "/workspace/src/main.py", "print('hello')")

        # First command should be mkdir -p for the parent directory
        assert any("mkdir -p" in cmd and "/workspace/src" in cmd for cmd in container.exec_commands)

    @pytest.mark.asyncio
    async def test_write_text_uses_base64_encoding(self):
        """Test write_text uses base64 to safely transfer content."""
        container = MockContainer()

        content = "def hello():\n    print('world')\n"
        cio = _make_container_io(container)
        await asyncio.to_thread(cio.write_text, "/workspace/main.py", content)

        # Find the base64 write command
        write_cmds = [cmd for cmd in container.exec_commands if "base64" in cmd]
        assert len(write_cmds) == 1

        # Verify the base64-encoded content is correct
        expected_b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        assert expected_b64 in write_cmds[0]

    @pytest.mark.asyncio
    async def test_write_text_handles_special_characters(self):
        """Test write_text safely handles content with quotes, backticks, and dollar signs."""
        container = MockContainer()

        content = "echo \"$HOME\" && `whoami` | tee 'output.txt'"
        cio = _make_container_io(container)
        await asyncio.to_thread(cio.write_text, "/workspace/script.sh", content)

        # Should complete without error (base64 avoids shell escaping issues)
        write_cmds = [cmd for cmd in container.exec_commands if "base64" in cmd]
        assert len(write_cmds) == 1

    @pytest.mark.asyncio
    async def test_write_text_dry_run(self):
        """Test write_text does nothing in dry_run mode."""
        container = MockContainer()

        cio = _make_container_io(container)
        cio.dry_run = True
        await asyncio.to_thread(cio.write_text, "/workspace/main.py", "content")

        # No commands should have been executed
        assert len(container.exec_commands) == 0

    @pytest.mark.asyncio
    async def test_write_text_file_in_root(self):
        """Test write_text handles a file without parent directory separator."""
        container = MockContainer()

        cio = _make_container_io(container)
        await asyncio.to_thread(cio.write_text, "file.txt", "content")

        # Should use "." as parent dir (shlex.quote(".") produces just ".")
        mkdir_cmds = [cmd for cmd in container.exec_commands if "mkdir -p" in cmd]
        assert len(mkdir_cmds) == 1
        assert "." in mkdir_cmds[0]


class TestContainerIOHelpers:
    """Test helper methods."""

    @pytest.mark.asyncio
    async def test_read_image_returns_none(self):
        """Test read_image always returns None (not supported)."""
        container = MockContainer()

        cio = _make_container_io(container)
        result = cio.read_image("/workspace/image.png")

        assert result is None

    @pytest.mark.asyncio
    async def test_confirm_ask_returns_true(self):
        """Test confirm_ask always returns True (auto-confirm)."""
        container = MockContainer()

        cio = _make_container_io(container)
        result = cio.confirm_ask("Are you sure?", default="n")

        assert result is True

    @pytest.mark.asyncio
    async def test_initialization_defaults(self):
        """Test ContainerIO sets correct defaults for non-interactive mode."""
        container = MockContainer()

        cio = _make_container_io(container)

        assert cio.yes is True
        assert cio.pretty is False
        assert cio.dry_run is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
