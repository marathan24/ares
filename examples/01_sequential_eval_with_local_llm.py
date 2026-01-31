"""Minimal example of using ARES with SWE-bench environment.

No API keys required - just local Docker containers and a local LLM.
It runs a few steps of the first task in SWE-bench Verified.

This example shows the basic way of interacting with an ARES environment
using two different code agents: MiniSWECodeAgent and Terminus2Agent.

This example uses Qwen2-0.5B-Instruct with a Llama CPP-backed LLM client.
Unfortunately this model is too weak to solve the task we've set it,
so we only run it for 5 steps with each agent.
In example 02_sequential_eval_with_api.py we'll use a more powerful LLM.

Prerequisites:

    - Install docker & make sure the daemon is running.
    - Install dependencies: `uv sync --group examples`
    - If you see Docker authentication errors (e.g., "email must be verified"):
        * RECOMMENDED: Set DOCKER_SKIP_AUTH=true to use anonymous pulls (no account needed)
        * Or run `docker logout` to clear stored credentials
        * Or verify your email at https://hub.docker.com/settings/general

Example usage:

    # Run with MiniSWECodeAgent (default)
    uv run -m examples.01_sequential_eval_with_local_llm --agent mswea

    # Run with Terminus2Agent
    uv run -m examples.01_sequential_eval_with_local_llm --agent terminus2

    # Run with both agents
    uv run -m examples.01_sequential_eval_with_local_llm --agent both
"""

import argparse
import asyncio

import ares
from ares.contrib import llama_cpp

from . import utils


async def run_episode(preset: str, agent, label: str) -> None:
    """Run a single episode with the given preset and agent."""
    print(f"\n{'=' * 80}")
    print(f"Running: {label} ({preset})")
    print(f"{'=' * 80}\n")

    # `:0` means load only the first task.
    # By default, ares.make will use local Docker containers.
    async with ares.make(preset) as env:
        # Reset the environment to get the first timestep
        ts = await env.reset()
        step_count = 0
        total_reward = 0.0

        while not ts.last():
            # The agent processes the observation and returns an action (LLM response)
            action = await agent(ts.observation)

            # Print the observation and action.
            utils.print_step(step_count, ts.observation, action)

            # Step the environment with the action
            ts = await env.step(action)

            assert ts.reward is not None
            total_reward += ts.reward
            step_count += 1

            # We only run for 5 steps; this model isn't strong enough to solve the task.
            if step_count >= 5:
                break

        # Episode complete!
        print()
        print("=" * 80)
        print(f"[{label}] Episode truncated after {step_count} steps")
        print(f"[{label}] Total reward: {total_reward}")
        print("=" * 80)


_AGENTS = {
    "mswea": ("sbv-mswea:0", "MiniSWECodeAgent"),
    "terminus2": ("sbv-terminus2:0", "Terminus2Agent"),
}


async def main(agent_choice: str) -> None:
    # Load Qwen2-0.5B-Instruct using a Llama CPP-backed LLM client.
    agent = llama_cpp.create_qwen2_0_5b_instruct_llama_cpp_client(n_ctx=32_768)

    agents_to_run = list(_AGENTS.values()) if agent_choice == "both" else [_AGENTS[agent_choice]]

    for preset, label in agents_to_run:
        await run_episode(preset, agent, label)

    print()
    print("🎉 You've seen ARES in action!")
    print()
    print("Next steps:")
    print("  - Try example 02_sequential_eval_with_api.py for a more powerful LLM")
    print("  - Try example 03_parallel_eval_with_api.py to evaluate an entire suite of tasks")
    print("  - Read the docs to learn more about ARES")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARES with a local LLM on SWE-bench Verified.")
    parser.add_argument(
        "--agent",
        choices=["mswea", "terminus2", "both"],
        default="mswea",
        help="Which code agent to use (default: mswea)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.agent))
