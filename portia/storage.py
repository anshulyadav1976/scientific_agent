"""Storage classes for managing the saving and retrieval of plans, runs, and tool calls.

This module defines a set of storage classes that provide different backends for saving, retrieving,
and managing plans, runs, and tool calls. These storage classes include both in-memory and
file-based storage, as well as integration with the Portia Cloud API. Each class is responsible
for handling interactions with its respective storage medium, including validating responses
and raising appropriate exceptions when necessary.

Classes:
    - Storage (Base Class): A base class that defines common interfaces for all storage types,
    ensuring consistent methods for saving and retrieving plans, runs, and tool calls.
    - InMemoryStorage: An in-memory implementation of the `Storage` class for storing plans,
    runs, and tool calls in a temporary, volatile storage medium.
    - FileStorage: A file-based implementation of the `Storage` class for storing plans, runs,
      and tool calls as local files in the filesystem.
    - PortiaCloudStorage: A cloud-based implementation of the `Storage` class that interacts with
    the Portia Cloud API to save and retrieve plans, runs, and tool call records.

Each storage class handles the following tasks:
    - Sending and receiving data to its respective storage medium - memory, file system, or API.
    - Validating responses from storage and raising errors when necessary.
    - Handling exceptions and re-raising them as custom `StorageError` exceptions to provide
    more informative error handling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, ValidationError

from portia.cloud import PortiaCloudClient
from portia.errors import PlanNotFoundError, PlanRunNotFoundError, StorageError
from portia.execution_agents.output import (
    AgentMemoryOutput,
    LocalOutput,
    Output,
)
from portia.execution_context import ExecutionContext
from portia.logger import logger
from portia.plan import Plan, PlanContext, PlanUUID, Step
from portia.plan_run import (
    PlanRun,
    PlanRunOutputs,
    PlanRunState,
    PlanRunUUID,
)
from portia.prefixed_uuid import PLAN_RUN_UUID_PREFIX
from portia.tool_call import ToolCallRecord, ToolCallStatus

if TYPE_CHECKING:
    from portia.config import Config

T = TypeVar("T", bound=BaseModel)

MAX_OUTPUT_LOG_LENGTH = 1000


class PlanStorage(ABC):
    """Abstract base class for storing and retrieving plans.

    Subclasses must implement the methods to save and retrieve plans.

    Methods:
        save_plan(self, plan: Plan) -> None:
            Save a plan.
        get_plan(self, plan_id: PlanUUID) -> Plan:
            Get a plan by ID.

    """

    @abstractmethod
    def save_plan(self, plan: Plan) -> None:
        """Save a plan.

        Args:
            plan (Plan): The Plan object to save.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("save_plan is not implemented")

    @abstractmethod
    def get_plan(self, plan_id: PlanUUID) -> Plan:
        """Retrieve a plan by its ID.

        Args:
            plan_id (PlanUUID): The UUID of the plan to retrieve.

        Returns:
            Plan: The Plan object associated with the provided plan_id.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("get_plan is not implemented")


class PlanRunListResponse(BaseModel):
    """Response for the get_plan_runs operation. Can support pagination."""

    results: list[PlanRun]
    count: int
    total_pages: int
    current_page: int


class RunStorage(ABC):
    """Abstract base class for storing and retrieving runs.

    Subclasses must implement the methods to save and retrieve PlanRuns.

    Methods:
        save_plan_run(self, run: Run) -> None:
            Save a PlanRun.
        get_plan_run(self, plan_run_id: PlanRunUUID) -> PlanRun:
            Get PlanRun by ID.
        get_plan_runs(self, run_state: RunState | None = None, page=int | None = None)
            -> PlanRunListResponse:
            Return runs that match the given run_state

    """

    @abstractmethod
    def save_plan_run(self, plan_run: PlanRun) -> None:
        """Save a PlanRun.

        Args:
            plan_run (PlanRun): The Run object to save.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("save_run is not implemented")

    @abstractmethod
    def get_plan_run(self, plan_run_id: PlanRunUUID) -> PlanRun:
        """Retrieve PlanRun by its ID.

        Args:
            plan_run_id (RunUUID): The UUID of the run to retrieve.

        Returns:
            Run: The Run object associated with the provided plan_run_id.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("get_run is not implemented")

    @abstractmethod
    def get_plan_runs(
        self,
        run_state: PlanRunState | None = None,
        page: int | None = None,
    ) -> PlanRunListResponse:
        """List runs by their state.

        Args:
            run_state (RunState | None): Optionally filter runs by their state.
            page (int | None): Optional pagination data

        Returns:
            list[Run]: A list of Run objects that match the given state.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("get_plan_runs is not implemented")


class AdditionalStorage(ABC):
    """Abstract base class for additional storage.

    Subclasses must implement the methods.

    Methods:
        save_tool_call(self, tool_call: ToolCallRecord) -> None:
            Save a tool_call.

    """

    @abstractmethod
    def save_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Save a ToolCall.

        Args:
            tool_call (ToolCallRecord): The ToolCallRecord object to save.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("save_tool_call is not implemented")


class LogAdditionalStorage(AdditionalStorage):
    """AdditionalStorage that logs calls rather than persisting them.

    Useful for storages that don't care about tool_calls etc.
    """

    def save_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Log the tool call.

        Args:
            tool_call (ToolCallRecord): The ToolCallRecord object to log.

        """
        logger().debug(
            f"Tool {tool_call.tool_name!s} executed in {tool_call.latency_seconds:.2f} seconds",
        )
        # Limit log to just first 1000 characters
        output = tool_call.output
        if len(str(tool_call.output)) > MAX_OUTPUT_LOG_LENGTH:
            output = (
                str(tool_call.output)[:MAX_OUTPUT_LOG_LENGTH]
                + "...[truncated - only first 1000 characters shown]"
            )
        match tool_call.status:
            case ToolCallStatus.SUCCESS:
                logger().debug(
                    f"Tool call {tool_call.tool_name!s} completed",
                    output=output,
                )
            case ToolCallStatus.FAILED:
                logger().error("Tool returned error", output=output)
            case ToolCallStatus.NEED_CLARIFICATION:
                logger().debug("Tool returned clarifications", output=output)


class Storage(PlanStorage, RunStorage, LogAdditionalStorage):
    """Combined base class for Plan Run + Additional storages."""


class AgentMemory(Protocol):
    """Abstract base class for storing items in agent memory."""

    @abstractmethod
    def save_plan_run_output(
        self,
        output_name: str,
        output: Output,
        plan_run_id: PlanRunUUID,
    ) -> Output:
        """Save an output from a plan run to agent memory.

        Args:
            output_name (str): The name of the output within the plan
            output (Output): The Output object to save
            plan_run_id (PlanRunUUID): The ID of the current plan run

        Returns:
            Output: The Output object with value marked as stored in agent memory.

        Raises:
            NotImplementedError: If the method is not implemented.

        """

    @abstractmethod
    def get_plan_run_output(self, output_name: str, plan_run_id: PlanRunUUID) -> Output:
        """Retrieve an Output from the storage.

        Args:
            output_name (str): The name of the output to retrieve
            plan_run_id (PlanRunUUID): The ID of the plan run

        Returns:
            Output: The retrieved Output object with value filled in from agent memory.

        Raises:
            NotImplementedError: If the method is not implemented.

        """


class InMemoryStorage(PlanStorage, RunStorage, LogAdditionalStorage, AgentMemory):
    """Simple storage class that keeps plans + runs in memory.

    Tool Calls are logged via the LogAdditionalStorage.
    """

    plans: dict[PlanUUID, Plan]
    runs: dict[PlanRunUUID, PlanRun]
    outputs: defaultdict[PlanRunUUID, dict[str, Output]]

    def __init__(self) -> None:
        """Initialize Storage."""
        self.plans = {}
        self.runs = {}
        self.outputs = defaultdict(dict)

    def save_plan(self, plan: Plan) -> None:
        """Add plan to dict.

        Args:
            plan (Plan): The Plan object to save.

        """
        self.plans[plan.id] = plan

    def get_plan(self, plan_id: PlanUUID) -> Plan:
        """Get plan from dict.

        Args:
            plan_id (PlanUUID): The UUID of the plan to retrieve.

        Returns:
            Plan: The Plan object associated with the provided plan_id.

        Raises:
            PlanNotFoundError: If the plan is not found.

        """
        if plan_id in self.plans:
            return self.plans[plan_id]
        raise PlanNotFoundError(plan_id)

    def save_plan_run(self, plan_run: PlanRun) -> None:
        """Add run to dict.

        Args:
            plan_run (PlanRun): The Run object to save.

        """
        self.runs[plan_run.id] = plan_run

    def get_plan_run(self, plan_run_id: PlanRunUUID) -> PlanRun:
        """Get run from dict.

        Args:
            plan_run_id (PlanRunUUID): The UUID of the PlanRun to retrieve.

        Returns:
            PlanRun: The PlanRun object associated with the provided plan_run_id.

        Raises:
            PlanRunNotFoundError: If the PlanRun is not found.

        """
        if plan_run_id in self.runs:
            return self.runs[plan_run_id]
        raise PlanRunNotFoundError(plan_run_id)

    def get_plan_runs(
        self,
        run_state: PlanRunState | None = None,
        page: int | None = None,  # noqa: ARG002
    ) -> PlanRunListResponse:
        """Get run from dict.

        Args:
            run_state (RunState | None): Optionally filter runs by their state.
            page (int | None): Optional pagination data which is not used for in memory storage.

        Returns:
            list[Run]: A list of Run objects that match the given state.

        """
        if not run_state:
            results = list(self.runs.values())
        else:
            results = [plan_run for plan_run in self.runs.values() if plan_run.state == run_state]

        return PlanRunListResponse(
            results=results,
            count=len(results),
            current_page=1,
            total_pages=1,
        )

    def save_plan_run_output(
        self,
        output_name: str,
        output: Output,
        plan_run_id: PlanRunUUID,
    ) -> Output:
        """Save Output from a plan run to memory.

        Args:
            output_name (str): The name of the output within the plan
            output (Output): The Output object to save
            plan_run_id (PlanRunUUID): The ID of the current plan run

        """
        if output.get_summary() is None:
            logger().warning(
                f"Storing Output {output} with no summary",
            )
        self.outputs[plan_run_id][output_name] = output
        return AgentMemoryOutput(
            output_name=output_name,
            plan_run_id=plan_run_id,
            summary=output.get_summary() or "",
        )

    def get_plan_run_output(self, output_name: str, plan_run_id: PlanRunUUID) -> Output:
        """Retrieve an Output from memory.

        Args:
            output_name (str): The name of the output to retrieve
            plan_run_id (PlanRunUUID): The ID of the plan run

        Returns:
            Output: The retrieved Output object

        Raises:
            KeyError: If the output is not found

        """
        return self.outputs[plan_run_id][output_name]


class DiskFileStorage(PlanStorage, RunStorage, LogAdditionalStorage, AgentMemory):
    """Disk-based implementation of the Storage interface.

    Stores serialized Plan and Run objects as JSON files on disk.
    """

    def __init__(self, storage_dir: str | None) -> None:
        """Set storage dir.

        Args:
            storage_dir (str | None): Optional directory for storing files.

        """
        self.storage_dir = storage_dir or ".portia"

    def _ensure_storage(self, file_path: str | None = None) -> None:
        """Ensure that we have the storage directories required.

        This ensures that the storage directory exists as well as any other sub-directories
        needed for the file_path.

        Raises:
            FileNotFoundError: If the directory cannot be created.

        """
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        if file_path:
            Path(self.storage_dir, file_path).parent.mkdir(parents=True, exist_ok=True)

    def _write(self, file_path: str, content: BaseModel) -> None:
        """Write a serialized Plan or Run to a JSON file.

        Args:
            file_path (str): Path of the file to write.
            content (BaseModel): The Plan or Run object to serialize.

        """
        self._ensure_storage(file_path)  # Ensure storage directory exists
        # Explicitly open with UTF-8 encoding to prevent UnicodeEncodeError on Windows
        with Path(self.storage_dir, file_path).open("w", encoding="utf-8") as file:
            file.write(content.model_dump_json(indent=4))

    def _read(self, file_name: str, model: type[T]) -> T:
        """Read a JSON file and deserialize it into a BaseModel instance.

        Args:
            file_name (str): Name of the file to read.
            model (type[T]): The model class to deserialize into.

        Returns:
            T: The deserialized model instance.

        Raises:
            FileNotFoundError: If the file is not found.
            ValidationError: If the deserialization fails.

        """
        # Explicitly open with UTF-8 encoding for reading
        with Path(self.storage_dir, file_name).open("r", encoding="utf-8") as file:
            f = file.read()
            return model.model_validate_json(f)

    def save_plan(self, plan: Plan) -> None:
        """Save a Plan object to the storage.

        Args:
            plan (Plan): The Plan object to save.

        """
        self._write(f"{plan.id}.json", plan)

    def get_plan(self, plan_id: PlanUUID) -> Plan:
        """Retrieve a Plan object by its ID.

        Args:
            plan_id (PlanUUID): The ID of the Plan to retrieve.

        Returns:
            Plan: The retrieved Plan object.

        Raises:
            PlanNotFoundError: If the Plan is not found or validation fails.

        """
        try:
            return self._read(f"{plan_id}.json", Plan)
        except (ValidationError, FileNotFoundError) as e:
            raise PlanNotFoundError(plan_id) from e

    def save_plan_run(self, plan_run: PlanRun) -> None:
        """Save PlanRun object to the storage.

        Args:
            plan_run (PlanRun): The Run object to save.

        """
        self._write(f"{plan_run.id}.json", plan_run)

    def get_plan_run(self, plan_run_id: PlanRunUUID) -> PlanRun:
        """Retrieve PlanRun object by its ID.

        Args:
            plan_run_id (RunUUID): The ID of the Run to retrieve.

        Returns:
            Run: The retrieved Run object.

        Raises:
            RunNotFoundError: If the Run is not found or validation fails.

        """
        try:
            return self._read(f"{plan_run_id}.json", PlanRun)
        except (ValidationError, FileNotFoundError) as e:
            raise PlanRunNotFoundError(plan_run_id) from e

    def get_plan_runs(
        self,
        run_state: PlanRunState | None = None,
        page: int | None = None,  # noqa: ARG002
    ) -> PlanRunListResponse:
        """Find all plan runs in storage that match state.

        Args:
            run_state (RunState | None): Optionally filter runs by their state.
            page (int | None): Optional pagination data which is not used for in memory storage.

        Returns:
            list[Run]: A list of Run objects that match the given state.

        """
        self._ensure_storage()

        plan_runs = []

        directory_path = Path(self.storage_dir)
        for f in directory_path.iterdir():
            if f.is_file() and f.name.startswith(PLAN_RUN_UUID_PREFIX):
                plan_run = self._read(f.name, PlanRun)
                if not run_state or plan_run.state == run_state:
                    plan_runs.append(plan_run)

        return PlanRunListResponse(
            results=plan_runs,
            count=len(plan_runs),
            current_page=1,
            total_pages=1,
        )

    def save_plan_run_output(
        self,
        output_name: str,
        output: Output,
        plan_run_id: PlanRunUUID,
    ) -> Output:
        """Save Output from a plan run to agent memory on disk.

        Args:
            output_name (str): The name of the output within the plan
            output (Output): The Output object to save
            plan_run_id (PlanRunUUID): The ID of the current plan run

        """
        filename = f"{plan_run_id}/{output_name}.json"
        self._write(filename, output)
        return AgentMemoryOutput(
            output_name=output_name,
            plan_run_id=plan_run_id,
            summary=output.get_summary() or "",
        )

    def get_plan_run_output(self, output_name: str, plan_run_id: PlanRunUUID) -> Output:
        """Retrieve an Output from agent memory on disk.

        Args:
            output_name (str): The name of the output to retrieve
            plan_run_id (PlanRunUUID): The ID of the plan run

        Returns:
            Output: The retrieved Output object

        Raises:
            FileNotFoundError: If the output file is not found
            ValidationError: If the deserialization fails

        """
        file_name = f"{plan_run_id}/{output_name}.json"
        return self._read(file_name, LocalOutput)


class PortiaCloudStorage(Storage, AgentMemory):
    """Save plans, runs and tool calls to portia cloud."""

    DEFAULT_MAX_CACHE_SIZE = 20

    def __init__(
        self,
        config: Config,
        cache_dir: str | None = None,
        max_cache_size: int = DEFAULT_MAX_CACHE_SIZE,
    ) -> None:
        """Initialize the PortiaCloudStorage instance.

        Args:
            config (Config): The configuration containing API details for Portia Cloud.
            cache_dir (str | None): Optional directory for local caching of outputs.
            max_cache_size (int): The maximum number of files to cache locally.

        """
        self.client = PortiaCloudClient().get_client(config)
        self.form_client = PortiaCloudClient().new_client(config, json_headers=False)
        self.cache_dir = cache_dir or ".portia/cache/agent_memory"
        self.max_cache_size = max_cache_size
        self._ensure_cache_dir()

    def _ensure_cache_dir(self, file_path: str | None = None) -> None:
        """Ensure that we have the cache directories required.

        This ensures that the cache directory exists as well as any other sub-directories
        needed for the file_path.

        Args:
            file_path (str | None): Optional path to ensure parent directories exist.

        """
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        if file_path:
            Path(self.cache_dir, file_path).parent.mkdir(parents=True, exist_ok=True)

    def _ensure_cache_size(self) -> None:
        """Manage the cache size by removing the oldest file if the cache is full."""
        json_files = list(Path(self.cache_dir).glob("**/*.json"))
        if len(json_files) >= self.max_cache_size:
            oldest_file = min(json_files, key=lambda f: f.stat().st_mtime)
            oldest_file.unlink()
            logger().debug(f"Removed oldest cache file: {oldest_file}")

    def _write_to_cache(self, file_path: str, content: BaseModel) -> None:
        """Write a serialized object to a JSON file in the cache.

        Args:
            file_path (str): Path of the file to write.
            content (BaseModel): The object to serialize.

        """
        self._ensure_cache_dir(file_path)
        self._ensure_cache_size()

        # Write the file
        with Path(self.cache_dir, file_path).open("w") as file:
            file.write(content.model_dump_json(indent=4))

    def _read_from_cache(self, file_name: str, model: type[T]) -> T:
        """Read a JSON file from cache and deserialize it into a BaseModel instance.

        Args:
            file_name (str): Name of the file to read.
            model (type[T]): The model class to deserialize into.

        Returns:
            T: The deserialized model instance.

        Raises:
            FileNotFoundError: If the file is not found.
            ValidationError: If the deserialization fails.

        """
        with Path(self.cache_dir, file_name).open("r") as file:
            f = file.read()
            return model.model_validate_json(f)

    def check_response(self, response: httpx.Response) -> None:
        """Validate the response from Portia API.

        Args:
            response (httpx.Response): The response from the Portia API to check.

        Raises:
            StorageError: If the response from the Portia API indicates an error.

        """
        if not response.is_success:
            error_str = str(response.content)
            logger().error(f"Error from Portia Cloud: {error_str}")
            raise StorageError(error_str)

    def save_plan(self, plan: Plan) -> None:
        """Save a plan to Portia Cloud.

        Args:
            plan (Plan): The Plan object to save to the cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        try:
            response = self.client.post(
                url="/api/v0/plans/",
                json={
                    "id": str(plan.id),
                    "query": plan.plan_context.query,
                    "tool_ids": plan.plan_context.tool_ids,
                    "steps": [step.model_dump(mode="json") for step in plan.steps],
                },
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)

    def get_plan(self, plan_id: PlanUUID) -> Plan:
        """Retrieve a plan from Portia Cloud.

        Args:
            plan_id (PlanUUID): The ID of the plan to retrieve.

        Returns:
            Plan: The Plan object retrieved from Portia Cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails or the plan does not exist.

        """
        try:
            response = self.client.get(
                url=f"/api/v0/plans/{plan_id}/",
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)
            response_json = response.json()
            return Plan(
                id=PlanUUID.from_string(response_json["id"]),
                plan_context=PlanContext(
                    query=response_json["query"],
                    tool_ids=response_json["tool_ids"],
                ),
                steps=[Step.model_validate(step) for step in response_json["steps"]],
            )

    def save_plan_run(self, plan_run: PlanRun) -> None:
        """Save PlanRun to Portia Cloud.

        Args:
            plan_run (PlanRun): The Run object to save to the cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        try:
            response = self.client.put(
                url=f"/api/v0/plan-runs/{plan_run.id}/",
                json={
                    "current_step_index": plan_run.current_step_index,
                    "state": plan_run.state,
                    "execution_context": plan_run.execution_context.model_dump(mode="json"),
                    "outputs": plan_run.outputs.model_dump(mode="json"),
                    "plan_id": str(plan_run.plan_id),
                },
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)

    def get_plan_run(self, plan_run_id: PlanRunUUID) -> PlanRun:
        """Retrieve PlanRun from Portia Cloud.

        Args:
            plan_run_id (RunUUID): The ID of the run to retrieve.

        Returns:
            Run: The Run object retrieved from Portia Cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails or the run does not exist.

        """
        try:
            response = self.client.get(
                url=f"/api/v0/plan-runs/{plan_run_id}/",
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)
            response_json = response.json()
            return PlanRun(
                id=PlanRunUUID.from_string(response_json["id"]),
                plan_id=PlanUUID.from_string(response_json["plan"]["id"]),
                current_step_index=response_json["current_step_index"],
                state=PlanRunState(response_json["state"]),
                execution_context=ExecutionContext.model_validate(
                    response_json["execution_context"],
                ),
                outputs=PlanRunOutputs.model_validate(response_json["outputs"]),
            )

    def get_plan_runs(
        self,
        run_state: PlanRunState | None = None,
        page: int | None = None,
    ) -> PlanRunListResponse:
        """Find all runs in storage that match state.

        Args:
            run_state (RunState | None): Optionally filter runs by their state.
            page (int | None): Optional pagination data which is not used for in memory storage.

        Returns:
            list[Run]: A list of Run objects retrieved from Portia Cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        try:
            query = {}
            if page:
                query["page"] = page
            if run_state:
                query["run_state"] = run_state.value
            response = self.client.get(
                url=f"/api/v0/plan-runs/?{urlencode(query)}",
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)
            response_json = response.json()
            return PlanRunListResponse(
                results=[
                    PlanRun(
                        id=PlanRunUUID.from_string(plan_run["id"]),
                        plan_id=PlanUUID.from_string(plan_run["plan"]["id"]),
                        current_step_index=plan_run["current_step_index"],
                        state=PlanRunState(plan_run["state"]),
                        execution_context=ExecutionContext.model_validate(
                            plan_run["execution_context"],
                        ),
                        outputs=PlanRunOutputs.model_validate(plan_run["outputs"]),
                    )
                    for plan_run in response_json["results"]
                ],
                count=response_json["count"],
                current_page=response_json["current_page"],
                total_pages=response_json["total_pages"],
            )

    def save_tool_call(self, tool_call: ToolCallRecord) -> None:
        """Save a tool call to Portia Cloud.

        Args:
            tool_call (ToolCallRecord): The ToolCallRecord object to save to the cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        try:
            response = self.client.post(
                url="/api/v0/tool-calls/",
                json={
                    "plan_run_id": str(tool_call.plan_run_id),
                    "tool_name": tool_call.tool_name,
                    "step": tool_call.step,
                    "end_user_id": tool_call.end_user_id or "",
                    "additional_data": tool_call.additional_data,
                    "input": tool_call.input,
                    "output": tool_call.output,
                    "status": tool_call.status,
                    "latency_seconds": tool_call.latency_seconds,
                },
            )
        except Exception as e:
            raise StorageError(e) from e
        else:
            self.check_response(response)
            LogAdditionalStorage.save_tool_call(self, tool_call)

    def save_plan_run_output(
        self,
        output_name: str,
        output: Output,
        plan_run_id: PlanRunUUID,
    ) -> Output:
        """Save Output from a plan run to Portia Cloud.

        Args:
            output_name (str): The name of the output within the plan
            output (Output): The Output object to save
            plan_run_id (PlanRun): The if of the current plan run

        Raises:
            StorageError: If the request to Portia Cloud fails.

        """
        try:
            response = self.form_client.put(
                url=f"/api/v0/agent-memory/plan-runs/{plan_run_id}/outputs/{output_name}/",
                files={
                    "value": (
                        "output",
                        BytesIO(output.serialize_value().encode("utf-8")),
                    ),
                },
                data={
                    "summary": output.get_summary(),
                },
            )
            self.check_response(response)

            # Save to local cache
            if isinstance(output, LocalOutput):
                cache_file_path = f"{plan_run_id}/{output_name}.json"
                self._write_to_cache(cache_file_path, output)
                logger().debug(f"Saved output to local cache: {cache_file_path}")

            return AgentMemoryOutput(
                output_name=output_name,
                plan_run_id=plan_run_id,
                summary=output.get_summary() or "",
            )
        except Exception as e:
            raise StorageError(e) from e

    def get_plan_run_output(self, output_name: str, plan_run_id: PlanRunUUID) -> Output:
        """Retrieve an Output from Portia Cloud.

        Args:
            output_name: The name of the output to get from memory
            plan_run_id (RunUUID): The ID of the run to retrieve.

        Returns:
            Run: The Run object retrieved from Portia Cloud.

        Raises:
            StorageError: If the request to Portia Cloud fails or the run does not exist.

        """
        # Try to get from local cache first
        cache_file_path = f"{plan_run_id}/{output_name}.json"
        try:
            return self._read_from_cache(cache_file_path, LocalOutput)
        except (FileNotFoundError, ValidationError):
            # If not in cache, fetch from Portia Cloud
            logger().debug(
                f"Output not found in local cache, fetching from Portia Cloud: {cache_file_path}",
            )

        try:
            # Retrieving a value is a two step process
            # 1. Get the output with the storage URL from the backend
            # 2. Fetch the value from the storage URL
            output_response = self.client.get(
                url=f"/api/v0/agent-memory/plan-runs/{plan_run_id}/outputs/{output_name}/",
            )
            self.check_response(output_response)
            output_json = output_response.json()
            summary = output_json["summary"]
            value_url = output_json["url"]

            with httpx.Client() as client:
                value_response = client.get(value_url)
                value_response.raise_for_status()

            # Create the output object
            output = LocalOutput(
                summary=summary,
                value=value_response.text,
            )

            # Save to local cache for future use
            self._write_to_cache(cache_file_path, output)
            logger().debug(f"Saved output to local cache: {cache_file_path}")
        except Exception as e:
            raise StorageError(e) from e
        else:
            return output
