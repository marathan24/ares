"""AiderAgent — wraps the Aider code agent for use within ARES's RL loop.

Implements the ARES CodeAgent protocol by running Aider's synchronous Coder in a
background thread, with LLM calls intercepted via AresModel and file I/O routed
through ContainerIO.
"""

import asyncio
import dataclasses
import logging

from ares.code_agents import code_agent_base
from ares.code_agents.aider_agent import ares_model as ares_model_mod
from ares.code_agents.aider_agent import container_io as container_io_mod
from ares.containers import containers
from ares.experiment_tracking import stat_tracker
from ares.llms import llm_clients

_LOGGER = logging.getLogger(__name__)


def _patch_aider_litellm_exceptions() -> None:
    """Register any new litellm exception types that aider doesn't know about yet.

    Aider's LiteLLMExceptions._load() iterates over all attributes in litellm
    ending in "Error" and raises if any are missing from its hardcoded list.
    When litellm adds new exception types (e.g. BadGatewayError), older aider
    versions crash even though ARES bypasses litellm for inference entirely.
    """
    import litellm

    from aider import exceptions as aider_exceptions

    for var in dir(litellm):
        if var.endswith("Error") and var not in aider_exceptions.LiteLLMExceptions.exception_info:
            ex_info = aider_exceptions.ExInfo(var, True, None)
            aider_exceptions.EXCEPTIONS.append(ex_info)
            aider_exceptions.LiteLLMExceptions.exception_info[var] = ex_info
            _LOGGER.debug("Patched aider exceptions with missing litellm type: %s", var)


_patch_aider_litellm_exceptions()


@dataclasses.dataclass(kw_only=True)
class AiderAgent(code_agent_base.CodeAgent):
    """Code agent that delegates to Aider for code editing within an ARES container.

    Aider runs synchronously in a thread. Its LLM calls are intercepted by AresModel
    (routed through the ARES LLMClient / RL loop) and its file I/O is routed through
    ContainerIO (which uses container.exec_run()).

    Attributes:
        container: ARES container (Docker or Daytona) where the code lives.
        llm_client: ARES LLMClient (typically QueueMediatedLLMClient for RL).
        tracker: Optional stat tracker for performance metrics.
        edit_format: Aider edit format — "diff", "whole", "udiff", etc.
        model_name: Model name for Aider's internal settings lookup.
        max_reflections: Max reflection loops for malformed-response recovery.
    """

    container: containers.Container
    llm_client: llm_clients.LLMClient
    tracker: stat_tracker.StatTracker = dataclasses.field(default_factory=stat_tracker.NullStatTracker)
    edit_format: str = "diff"
    model_name: str = "gpt-4o"
    max_reflections: int = 3
    max_turns: int = 50

    async def run(self, task: str) -> None:
        """Run Aider on the given task inside the container.

        Uses Aider's interactive loop (Path B) instead of single-message mode.
        ContainerIO.get_input() feeds the initial task and subsequent container
        observations, keeping the loop alive across multiple RL steps — each
        send_completion() call goes through QueueMediatedLLMClient as a proper
        RL observation/action pair.
        """
        event_loop = asyncio.get_running_loop()

        _LOGGER.info("[%d] Starting AiderAgent run.", id(self))

        # Determine the container's working directory.
        with self.tracker.timeit("aider/setup"):
            workdir_result = await self.container.exec_run("pwd")
            container_root = workdir_result.output.strip()
            _LOGGER.debug("[%d] Container root: %s", id(self), container_root)

        # Create the LLM bridge.
        model = ares_model_mod.AresModel(
            model_name=self.model_name,
            ares_llm_client=self.llm_client,
            event_loop=event_loop,
        )

        # Create the file I/O bridge with the initial task and turn limit.
        # get_input() will return the task on the first call, then container
        # observations on subsequent calls, and raise EOFError at max_turns.
        io = container_io_mod.ContainerIO(
            container=self.container,
            event_loop=event_loop,
            initial_message=task,
            max_turns=self.max_turns,
        )

        def _create_and_run_coder() -> None:
            """Synchronous function that runs in a thread — creates and runs the Aider Coder."""
            from aider.coders import Coder

            coder = Coder.create(
                main_model=model,
                edit_format=self.edit_format,
                io=io,
                # Don't pass fnames — container files don't exist on the local filesystem,
                # so Coder.__init__'s Path(fname).exists() checks would fail.
                # Aider will discover files from the LLM's edit instructions.
                fnames=[],
                use_git=False,
                auto_commits=False,
                stream=False,
                auto_lint=False,
                auto_test=False,
                suggest_shell_commands=False,
                map_tokens=0,  # Disable repo map (needs local filesystem + tree-sitter).
                detect_urls=False,
            )

            # Override root to the container's working directory so that Aider resolves
            # relative paths against the container filesystem (via ContainerIO).
            coder.root = container_root
            coder.max_reflections = self.max_reflections

            _LOGGER.debug("[AiderAgent] Running Coder in interactive mode with task (%d chars)", len(task))
            # Use interactive mode (no with_message) so Aider enters its while-True
            # loop, calling get_input() each iteration. This keeps the agent alive
            # across multiple RL steps instead of returning after one LLM call.
            coder.run()

        with self.tracker.timeit("aider/run"):
            await asyncio.to_thread(_create_and_run_coder)

        _LOGGER.info("[%d] AiderAgent run complete.", id(self))
