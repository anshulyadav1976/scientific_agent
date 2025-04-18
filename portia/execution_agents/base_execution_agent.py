"""Agents are responsible for executing steps of a PlanRun.

The BaseAgent class is the base class that all agents must extend.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from portia.execution_agents.context import build_context

# TODO: Remove this once the backend / evals are updated to use the new import  # noqa: FIX002, TD002, TD003, E501
from portia.execution_agents.output import Output  # noqa: TC001
from portia.execution_context import get_execution_context

if TYPE_CHECKING:
    from portia.config import Config
    from portia.plan import Step
    from portia.plan_run import PlanRun
    from portia.tool import Tool


class BaseExecutionAgent:
    """An ExecutionAgent is responsible for carrying out the task defined in the given Step.

    This BaseExecutionAgent is the class all ExecutionAgents must extend. Critically,
    ExecutionAgents must implement the execute_sync function which is responsible for
    actually carrying out the task as given in the step. They have access to copies of the
    step, plan_run and config but changes to those objects are forbidden.

    Optionally, new execution agents may also override the get_context function, which is
    responsible for building the system context for the agent. This should be done with
    thought, as the details of the system context are critically important for LLM
    performance.
    """

    def __init__(
        self,
        step: Step,
        plan_run: PlanRun,
        config: Config,
        tool: Tool | None = None,
    ) -> None:
        """Initialize the base agent with the given args.

        Importantly, the models here are frozen copies of those used by the Portia instance.
        They are meant as read-only references, useful for execution of the task
        but cannot be edited. The agent should return output via the response
        of the execute_sync method.

        Args:
            step (Step): The step that defines the task to be executed.
            plan_run (PlanRun): The run that contains the step and related data.
            config (Config): The configuration settings for the agent.
            tool (Tool | None): An optional tool associated with the agent (default is None).

        """
        self.step = step
        self.tool = tool
        self.config = config
        self.plan_run = plan_run

    @abstractmethod
    def execute_sync(self) -> Output:
        """Run the core execution logic of the task synchronously.

        Implementation of this function is deferred to individual agent implementations,
        making it simple to write new ones.

        Returns:
            Output: The output of the task execution.

        """

    def get_system_context(self) -> str:
        """Build a generic system context string from the step and run provided.

        This function retrieves the execution context and generates a system context
        based on the step and run provided to the agent.

        Returns:
            str: A string containing the system context for the agent.

        """
        ctx = get_execution_context()
        return build_context(
            ctx,
            self.step,
            self.plan_run,
        )
