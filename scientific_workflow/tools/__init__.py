"""Custom tools for the Scientific Workflow Agent."""

# Import specific tools into this namespace for easy registration
# Comment out unused tools for the new Ollama-centric approach
# from .data_tools import DataIngestionTool
# from .analysis_tools import AnalysisTool
# Import the new Ollama tool
from .ollama_tool import OllamaTool
# Import SearchTool if/when needed
# from .search_tools import ScientificSearchTool

from portia.tool_registry import InMemoryToolRegistry
# Import Portia's built-in tools if needed directly
from portia.open_source_tools.registry import open_source_tool_registry # Contains SearchTool, LLMTool etc.


def create_tool_registry() -> InMemoryToolRegistry:
    """Creates and returns the tool registry for the agent.

    Includes both custom tools and potentially relevant built-in tools.
    """
    custom_tools = [
        # Add instances of custom tools here as they are created
        OllamaTool(),
        # Add SearchTool instance here later if needed
    ]

    # Combine custom tools with relevant open-source tools
    # We probably don't need LLMTool if OllamaTool handles generation.
    # Keep SearchTool for M3.
    existing_custom_ids = {tool.id for tool in custom_tools}
    relevant_oss_tools = ["search_tool"] # Keep potentially needed OSS tools
    combined_tools = custom_tools + [
        tool for tool in open_source_tool_registry.get_tools()
        if tool.id in relevant_oss_tools and tool.id not in existing_custom_ids
    ]

    # It's important that tool IDs are unique
    tool_ids = [tool.id for tool in combined_tools]
    if len(tool_ids) != len(set(tool_ids)):
        # Provide more detail in the error
        counts = {id: tool_ids.count(id) for id in tool_ids}
        duplicates = {id: count for id, count in counts.items() if count > 1}
        raise ValueError(f"Duplicate tool IDs found in the registry! Duplicates: {duplicates}")


    return InMemoryToolRegistry.from_local_tools(combined_tools) 