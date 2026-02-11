"""ContainerIO — bridges Aider's local filesystem I/O to ARES container operations.

Subclasses Aider's InputOutput and overrides read_text() / write_text() so that all
file operations performed by Aider coders (reading source files, writing edits) are
routed through the ARES Container protocol via exec_run().

Also overrides get_input() to enable multi-step RL interaction: instead of reading
from a terminal prompt, get_input() checks the container state (e.g. git diff) and
feeds observations back as the next "user message", keeping Aider's interactive
while-loop alive across multiple RL steps.
"""

import asyncio
import base64
import logging
import shlex
from typing import Any

from aider.io import InputOutput

from ares.containers import containers

_LOGGER = logging.getLogger(__name__)

# Maximum characters of git diff / test output to include in feedback.
_MAX_OBSERVATION_CHARS = 3000


class ContainerIO(InputOutput):
    """Aider InputOutput subclass that routes file I/O through an ARES container.

    All file paths received by read_text/write_text are container-absolute paths
    (e.g. /workspace/src/main.py). The container's exec_run() is used to read/write
    files inside the container.

    For multi-step RL interaction, get_input() is overridden to return container
    observations instead of blocking on terminal input. This keeps Aider's
    interactive while-loop alive so each LLM call becomes a proper RL step.
    """

    def __init__(
        self,
        container: containers.Container,
        event_loop: asyncio.AbstractEventLoop,
        *,
        initial_message: str | None = None,
        max_turns: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            yes=True,  # Non-interactive: auto-confirm everything.
            pretty=False,  # No rich formatting.
            fancy_input=False,  # No prompt_toolkit input.
            dry_run=False,
            **kwargs,
        )
        self._container = container
        self._event_loop = event_loop
        self._initial_message = initial_message
        self._max_turns = max_turns
        self._turn_count = 0

    def _run_in_container(self, command: str) -> containers.ExecResult:
        """Execute a command in the container from Aider's sync thread."""
        future = asyncio.run_coroutine_threadsafe(
            self._container.exec_run(command),
            self._event_loop,
        )
        return future.result()

    def read_text(self, filename, silent=False):
        """Read a file from the container instead of the local filesystem."""
        filename = str(filename)

        try:
            result = self._run_in_container(f"cat {shlex.quote(filename)}")
        except Exception as e:
            if not silent:
                self.tool_error(f"{filename}: unable to read from container: {e}")
            return None

        if result.exit_code != 0:
            if not silent:
                _LOGGER.debug("[ContainerIO] read_text failed for %s: %s", filename, result.output.strip())
            return None

        return result.output

    def write_text(self, filename, content, max_retries=5, initial_delay=0.1):
        """Write a file into the container instead of the local filesystem.

        Uses base64 encoding to safely transfer arbitrary file content through
        the container's exec_run() interface, avoiding shell escaping issues with
        special characters, quotes, and newlines.
        """
        del max_retries, initial_delay  # Not applicable for container writes.

        if self.dry_run:
            return

        filename = str(filename)

        # Ensure the parent directory exists in the container.
        parent_dir = filename.rsplit("/", 1)[0] if "/" in filename else "."
        self._run_in_container(f"mkdir -p {shlex.quote(parent_dir)}")

        # Base64-encode the content to avoid shell escaping issues.
        encoded = base64.b64encode(content.encode(self.encoding)).decode("ascii")
        result = self._run_in_container(f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(filename)}")

        if result.exit_code != 0:
            self.tool_error(f"{filename}: unable to write to container: {result.output.strip()}")
            return

        _LOGGER.debug("[ContainerIO] Wrote %d bytes to %s", len(content), filename)

    def read_image(self, filename):  # noqa: ARG002
        """Image support disabled for container-based execution."""
        return None

    def confirm_ask(self, *args, **kwargs):  # noqa: ARG002
        """Auto-confirm all prompts in non-interactive mode."""
        return True

    def get_input(
        self,
        root=None,  # noqa: ARG002
        rel_fnames=None,  # noqa: ARG002
        addable_rel_fnames=None,  # noqa: ARG002
        commands=None,  # noqa: ARG002
        abs_read_only_fnames=None,  # noqa: ARG002
        edit_format=None,  # noqa: ARG002
    ):
        """Override Aider's interactive input with container-aware observation feedback.

        On the first call, returns the initial task message. On subsequent calls,
        inspects the container state (git diff) and returns observations as the next
        "user message" for Aider to act on — mirroring how Terminus2 feeds command
        output back into its conversation loop.

        Raises EOFError when max_turns is reached, which Aider's run() loop catches
        to cleanly terminate the agent.
        """
        self._turn_count += 1

        if self._turn_count > self._max_turns:
            _LOGGER.info("[ContainerIO] Reached max turns (%d). Signaling EOF.", self._max_turns)
            raise EOFError

        # First turn: return the initial task.
        if self._turn_count == 1 and self._initial_message:
            _LOGGER.debug("[ContainerIO] Turn 1: returning initial task message.")
            self.user_input(self._initial_message)
            return self._initial_message

        # Subsequent turns: check container state and provide feedback.
        observation = self._build_observation()

        _LOGGER.debug("[ContainerIO] Turn %d: returning observation (%d chars).", self._turn_count, len(observation))
        self.user_input(observation)
        return observation

    def _build_observation(self) -> str:
        """Inspect the container and build a feedback message for Aider.

        Checks git diff to see what has changed, similar to how Terminus2 executes
        commands and feeds the output back as the next user message.
        """
        parts: list[str] = []

        # Check what files were modified.
        try:
            diff_result = self._run_in_container("git diff --stat")
            if diff_result.exit_code == 0 and diff_result.output.strip():
                parts.append("Files changed so far:")
                parts.append(diff_result.output.strip()[:_MAX_OBSERVATION_CHARS])
            else:
                parts.append("No file changes detected from your previous response.")
                parts.append("Please try a different approach to fix the issue.")
                return "\n".join(parts)
        except Exception:
            _LOGGER.debug("[ContainerIO] git diff failed, skipping.")

        # Show the actual diff content for context.
        try:
            full_diff = self._run_in_container("git diff")
            if full_diff.exit_code == 0 and full_diff.output.strip():
                diff_text = full_diff.output.strip()[:_MAX_OBSERVATION_CHARS]
                parts.append("")
                parts.append("Diff:")
                parts.append(f"```\n{diff_text}\n```")
        except Exception:
            _LOGGER.debug("[ContainerIO] git diff (full) failed, skipping.")

        parts.append("")
        parts.append("Please review the changes and continue working on the task if more edits are needed.")

        return "\n".join(parts)
