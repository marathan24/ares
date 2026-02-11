"""Tests for AiderAgent — the main agent orchestrator.

Tests agent initialization, default field values, container working directory
detection, and the Coder creation/execution flow with mocked Aider internals.
"""

from unittest import mock

import pytest

from ares.code_agents.aider_agent import aider_agent
from ares.containers import containers
from ares.experiment_tracking import stat_tracker
from ares.llms import llm_clients
from ares.testing.mock_container import MockContainer


class TestAiderAgentBasics:
    """Test basic agent construction and defaults."""

    def test_default_field_values(self):
        """Test agent initializes with correct default attributes."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = aider_agent.AiderAgent(
            container=container,
            llm_client=llm_client,
        )

        assert agent.edit_format == "diff"
        assert agent.model_name == "gpt-4o"
        assert agent.max_reflections == 3
        assert isinstance(agent.tracker, stat_tracker.NullStatTracker)

    def test_custom_field_values(self):
        """Test agent accepts custom configuration."""
        container = MockContainer()
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = aider_agent.AiderAgent(
            container=container,
            llm_client=llm_client,
            edit_format="whole",
            model_name="claude-3-5-sonnet-20241022",
            max_reflections=5,
        )

        assert agent.edit_format == "whole"
        assert agent.model_name == "claude-3-5-sonnet-20241022"
        assert agent.max_reflections == 5


class TestAiderAgentRun:
    """Test the run() method with mocked Aider Coder."""

    @pytest.mark.asyncio
    async def test_run_queries_container_workdir(self):
        """Test run() queries the container's working directory."""
        container = MockContainer(exec_responses={"pwd": containers.ExecResult(output="/testbed\n", exit_code=0)})
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = aider_agent.AiderAgent(
            container=container,
            llm_client=llm_client,
        )

        with mock.patch("ares.code_agents.aider_agent.aider_agent.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = None
            await agent.run("Fix the bug")

        # Verify pwd was called
        assert "pwd" in container.exec_commands

    @pytest.mark.asyncio
    async def test_run_creates_coder_with_correct_args(self):
        """Test run() creates Aider Coder with expected configuration."""
        container = MockContainer(exec_responses={"pwd": containers.ExecResult(output="/testbed\n", exit_code=0)})
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = aider_agent.AiderAgent(
            container=container,
            llm_client=llm_client,
            edit_format="udiff",
            max_reflections=7,
        )

        coder_mock = mock.MagicMock()

        with mock.patch("aider.coders.Coder.create", return_value=coder_mock) as mock_create:
            # We need to actually run the closure in-thread, so mock to_thread
            # to call the function directly (synchronously).
            async def run_sync(fn, *args, **kwargs):
                return fn(*args, **kwargs)

            with mock.patch("ares.code_agents.aider_agent.aider_agent.asyncio.to_thread", side_effect=run_sync):
                await agent.run("Fix the bug")

        # Verify Coder.create was called with correct args
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["edit_format"] == "udiff"
        assert call_kwargs["fnames"] == []
        assert call_kwargs["use_git"] is False
        assert call_kwargs["auto_commits"] is False
        assert call_kwargs["stream"] is False
        assert call_kwargs["auto_lint"] is False
        assert call_kwargs["auto_test"] is False
        assert call_kwargs["suggest_shell_commands"] is False
        assert call_kwargs["map_tokens"] == 0
        assert call_kwargs["detect_urls"] is False

        # Verify coder.root was set to container workdir
        assert coder_mock.root == "/testbed"
        assert coder_mock.max_reflections == 7

        # Verify coder.run was called in interactive mode (no with_message).
        # The task is fed through ContainerIO.get_input() instead.
        coder_mock.run.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_run_strips_workdir_whitespace(self):
        """Test run() strips trailing whitespace/newlines from pwd output."""
        container = MockContainer(exec_responses={"pwd": containers.ExecResult(output="  /workspace  \n", exit_code=0)})
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = aider_agent.AiderAgent(
            container=container,
            llm_client=llm_client,
        )

        coder_mock = mock.MagicMock()

        with mock.patch("aider.coders.Coder.create", return_value=coder_mock):

            async def run_sync(fn, *args, **kwargs):
                return fn(*args, **kwargs)

            with mock.patch("ares.code_agents.aider_agent.aider_agent.asyncio.to_thread", side_effect=run_sync):
                await agent.run("task")

        assert coder_mock.root == "/workspace"

    @pytest.mark.asyncio
    async def test_run_passes_model_to_coder(self):
        """Test run() creates an AresModel and passes it to Coder.create."""
        container = MockContainer(exec_responses={"pwd": containers.ExecResult(output="/testbed\n", exit_code=0)})
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = aider_agent.AiderAgent(
            container=container,
            llm_client=llm_client,
            model_name="gpt-4-turbo",
        )

        coder_mock = mock.MagicMock()

        with mock.patch("aider.coders.Coder.create", return_value=coder_mock) as mock_create:

            async def run_sync(fn, *args, **kwargs):
                return fn(*args, **kwargs)

            with mock.patch("ares.code_agents.aider_agent.aider_agent.asyncio.to_thread", side_effect=run_sync):
                await agent.run("task")

        # Verify the model passed to Coder.create is an AresModel
        from ares.code_agents.aider_agent import ares_model

        passed_model = mock_create.call_args[1]["main_model"]
        assert isinstance(passed_model, ares_model.AresModel)
        assert passed_model.name == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_run_passes_container_io_to_coder(self):
        """Test run() creates a ContainerIO and passes it to Coder.create."""
        container = MockContainer(exec_responses={"pwd": containers.ExecResult(output="/testbed\n", exit_code=0)})
        llm_client = mock.AsyncMock(spec=llm_clients.LLMClient)

        agent = aider_agent.AiderAgent(
            container=container,
            llm_client=llm_client,
        )

        coder_mock = mock.MagicMock()

        with mock.patch("aider.coders.Coder.create", return_value=coder_mock) as mock_create:

            async def run_sync(fn, *args, **kwargs):
                return fn(*args, **kwargs)

            with mock.patch("ares.code_agents.aider_agent.aider_agent.asyncio.to_thread", side_effect=run_sync):
                await agent.run("task")

        # Verify the io passed to Coder.create is a ContainerIO
        from ares.code_agents.aider_agent import container_io

        passed_io = mock_create.call_args[1]["io"]
        assert isinstance(passed_io, container_io.ContainerIO)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
