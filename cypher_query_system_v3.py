import os
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from py2neo import Graph, Node, Relationship, Path

# --- PORTIA IMPORTS ---
try:
    from portia import (
        Portia,
        Plan,
        PlanRun,
        PlanRunState,
        default_config,
        StorageClass
    )
    # Initialize default Portia config (using memory storage)
    portia_config = default_config()
    portia_config.storage_class = StorageClass.MEMORY
    PORTIA_AVAILABLE = True
    print("Portia SDK imported successfully. Using MEMORY storage.")
except ImportError as e:
    print(f"Warning: Portia SDK not found ({e}). Running without Portia integration.")
    PORTIA_AVAILABLE = False
    # Define dummy classes if Portia is not available, so the rest of the code doesn't break
    class PlanRunState:
        COMPLETE = "COMPLETE"
        FAILED = "FAILED"
        IN_PROGRESS = "IN_PROGRESS"
        NOT_STARTED = "NOT_STARTED"
# --- END PORTIA IMPORTS ---


def load_string_from_file(filepath):
  """Reads the entire content of a text file into a string."""
  try:
    with open(filepath, 'r', encoding='utf-8') as f:
      content = f.read()
    return content
  except FileNotFoundError:
    print(f"Error: The file '{filepath}' was not found.")
    return None
  except Exception as e:
    print(f"An error occurred: {e}")
    return None

# Import py2neo types at the top of the file if not already done
from py2neo import Node, Relationship, Path

# --- Helper Function to safely convert property values to JSON serializable types ---
def serialize_property(value):
    if isinstance(value, (Node, Relationship, Path)):
        # Convert graph objects to a simple string representation
        return str(value)
    elif isinstance(value, list):
        # Recursively serialize items in a list
        return [serialize_property(item) for item in value]
    elif isinstance(value, dict):
         # Recursively serialize values in a dict
         return {k: serialize_property(v) for k, v in value.items()}
    elif isinstance(value, (str, int, float, bool, type(None))):
        # Return basic types directly
        return value
    else:
        # Fallback for other complex types
        return str(value)


# Create a single mapping dictionary for prompt formatting
PROMPT_PLACEHOLDERS = {
    "schema": "self.schema",  # The full schema object
    "schema_desc": "self.schema_desc",  # Natural language description of schema
    "db_mappings_text": "self.db_mappings_text",  # Database mappings information
}


# Define system prompts as constants with descriptive docstrings

# Prompt for handling schema validation
SCHEMA_VALIDATION_PROMPT = load_string_from_file('prompts/schema_validation_prompt.txt')

"""System prompt used for schema validation to identify when user terminology doesn't match the database schema.
This prompt instructs the LLM to analyze user queries and detect terminology mismatches with available node types."""

# Prompt for generating Cypher queries
CYPHER_GENERATION_PROMPT = load_string_from_file('prompts/cypher_generation_prompt_common.txt')

"""System prompt for the query generation agent to translate natural language to Cypher queries.
This prompt includes slots for database schema description, database mappings information, and 
optional verification feedback when refining a query."""

# Prompt for verifying Cypher queries
QUERY_VERIFICATION_PROMPT = load_string_from_file('prompts/query_verification_prompt.txt')

"""System prompt for the verification agent to check if generated Cypher queries match the user's intent.
This prompt helps verify that the translation from natural language to Cypher correctly captures the user's request."""

# Prompt for error correction
ERROR_CORRECTION_PROMPT = load_string_from_file('prompts/error_correction_prompt.txt')

"""System prompt for the error correction step when a query fails during execution.
This prompt guides the LLM to fix specific syntax or schema-related issues in the Cypher query."""

# Prompt for query reformulation
QUERY_REFORMULATION_PROMPT = load_string_from_file('prompts/query_reformulation_prompt.txt')

"""System prompt for the complete query reformulation step when initial correction attempts fail.
This prompt instructs the more powerful model to completely rethink the approach to the query."""

# Prompt for alternative suggestions
ALTERNATIVE_SUGGESTION_PROMPT = load_string_from_file('prompts/alternative_suggestion_prompt_common.txt')

"""System prompt for suggesting alternative approaches when all query attempts have failed.
This prompt helps provide useful fallback options to users when their original request cannot be fulfilled."""

# Prompt for entity name resolution
ENTITY_RESOLUTION_PROMPT = load_string_from_file('prompts/entity_resolution_prompt.txt')

"""System prompt for resolving entity identifiers to common names based on the model's knowledge.
This prompt helps translate database-specific identifiers to human-readable entity names."""

# Prompt for natural language response generation
NATURAL_LANGUAGE_RESPONSE_PROMPT = load_string_from_file('prompts/natural_language_response_prompt.txt')

# Prompt for database source identification
DATABASE_SOURCE_PROMPT = load_string_from_file('prompts/database_source_prompt.txt')

"""Prompt for identifying the likely source databases for entity identifiers in the graph.
This helps map IDs to their source databases for better query construction and entity resolution."""

class LLMWrapper:
    """
    A wrapper class for interacting with various LLM APIs.
    Currently supports OpenAI and Google (Gemini) providers.
    """
    
    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ):
        """
        Initialize the LLM wrapper.
        
        Args:
            provider: The LLM provider ('openai' or 'google')
            model: The specific model to use (e.g., 'gpt-4', 'gemini-pro')
            api_key: API key for the provider (optional, will look for env var if not provided)
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Penalizes frequent tokens
            presence_penalty: Penalizes tokens already present
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        
        # Validate provider
        if self.provider not in ['openai', 'google']:
            raise ValueError("Provider must be 'openai' or 'google'")
        
        # Set up API key
        if api_key:
            self.api_key = api_key
        else:
            # Try to get API key from environment variables
            if self.provider == 'openai':
                self.api_key = os.environ.get('OPENAI_API_KEY')
            elif self.provider == 'google':
                self.api_key = os.environ.get('GOOGLE_API_KEY')
            
            if not self.api_key:
                raise ValueError(f"API key for {self.provider} not provided and not found in environment variables")
    
    def get_output(self, prompt: Union[str, List[Dict[str, str]]], system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt to the LLM and get the response.
        
        Args:
            prompt: The prompt to send to the LLM. Can be a simple string or a list of message dictionaries
                    for chat-based models
            system_prompt: Optional system prompt for models that support it
            
        Returns:
            The LLM's response as a string
        """
        if self.provider == 'openai':
            return self._get_openai_output(prompt, system_prompt)
        elif self.provider == 'google':
            return self._get_google_output(prompt, system_prompt)
    
    def _get_openai_output(self, prompt: Union[str, List[Dict[str, str]]], system_prompt: Optional[str] = None) -> str:
        """
        Send a prompt to OpenAI's API and get the response using the new client format.
        """
        from openai import OpenAI
        
        # Initialize the client
        client = OpenAI(api_key=self.api_key)
        
        # Check if we're dealing with a chat model
        if "gpt" in self.model.lower():
            # For the new GPT models that use the messages API
            
            # Handle different prompt formats
            if isinstance(prompt, str):
                # If prompt is a string, we need to format it properly
                if system_prompt:
                    # If using system_prompt parameter with the new responses API
                    response = client.responses.create(
                        model=self.model,
                        instructions=system_prompt,
                        input=prompt,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,  # Changed parameter name
                        top_p=self.top_p
                    )
                    return response.output_text.strip()
                else:
                    # No system prompt, using chat completions API with messages
                    messages = [{"role": "user", "content": prompt}]
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,  # This API still uses max_tokens
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty
                    )
                    return response.choices[0].message.content.strip()
            else:
                # Prompt is already a list of messages
                # Add system message if provided and not already included
                messages = prompt.copy()  # Create a copy to avoid modifying the original
                if system_prompt and not any(m.get("role") == "system" for m in messages):
                    messages.insert(0, {"role": "system", "content": system_prompt})
                
                # Use chat completions API
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty
                )
                return response.choices[0].message.content.strip()
        else:
            # For legacy completion models
            response = client.completions.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            return response.choices[0].text.strip()
    
    def _get_google_output(
        self, 
        prompt: Union[str, List[Dict[str, str]]], 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Send a prompt to Google's Gemini API and get the response.
        """
        from google import genai
        from google.genai import types
        
        # Initialize the client
        client = genai.Client(api_key=self.api_key)
        
        # Prepare contents based on prompt type and system prompt
        contents = []
        
        # Process the main prompt
        if isinstance(prompt, str):
            # Single string prompt - prepend system prompt if it exists
            content_text = prompt
            if system_prompt:
                content_text = f"{system_prompt}\n\n{prompt}"
                
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content_text)]
                )
            )
 
        else:
            # List of message dictionaries
            # For message list format, need to integrate system prompt differently
            first_user_msg_found = False
            for message in prompt:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                # For Google, we don't use "system" role - convert to user content
                if role == "system":
                    role = "user"
                
                # If this is the first user message and we have a system prompt, prepend it
                if role == "user" and not first_user_msg_found and system_prompt:
                    content = f"{system_prompt}\n\n{content}"
                    first_user_msg_found = True
                    
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=content)]
                    )
                )
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            response_mime_type="text/plain",
        )
        
        # Generate content
        response = client.models.generate_content(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Extract and return the text
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.text
        elif hasattr(response, 'parts') and response.parts:
            return response.parts[0].text
        else:
            return ""

    def stream_output(self, prompt: Union[str, List[Dict[str, str]]], system_prompt: Optional[str] = None) -> Any:
        """
        Stream the response from the LLM.
        Returns a generator or stream object depending on the provider.
        
        Args:
            prompt: The prompt to send to the LLM
            system_prompt: Optional system prompt
            
        Returns:
            A generator of response chunks
        """
        if self.provider == 'openai':
            return self._stream_openai_output(prompt, system_prompt)
        elif self.provider == 'google':
            return self._stream_google_output(prompt, system_prompt)
    
    def _stream_openai_output(self, prompt: Union[str, List[Dict[str, str]]], system_prompt: Optional[str] = None) -> Any:
        """Stream response from OpenAI using the new client format"""
        from openai import OpenAI
        
        # Initialize the client
        client = OpenAI(api_key=self.api_key)
        
        # Handle different prompt formats
        if isinstance(prompt, str):
            if "gpt" in self.model.lower():
                # For chat models with string prompt
                if system_prompt:
                    # Using the responses API with instructions/input
                    return client.responses.create(
                        model=self.model,
                        instructions=system_prompt,
                        input=prompt,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,  # Note different parameter name
                        top_p=self.top_p,
                        stream=True
                    )
                else:
                    # No system prompt, using chat completions with message format
                    messages = [{"role": "user", "content": prompt}]
                    return client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        stream=True
                    )
            else:
                # For older completion models
                return client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                    stream=True
                )
        else:
            # Prompt is already a list of messages
            messages = prompt.copy()
            if system_prompt and not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            # Stream the response using chat completions API
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stream=True
            )

    def _stream_google_output(
        self, 
        prompt: Union[str, List[Dict[str, str]]], 
        system_prompt: Optional[str] = None
    ) -> Any:
        """
        Stream response from Google Gemini API.
        """
        from google import genai
        from google.genai import types
        
        # Initialize the client
        client = genai.Client(api_key=self.api_key)
        
        # Prepare contents based on prompt type and system prompt
        contents = []
        
        # Add system prompt if provided
        if system_prompt:
            contents.append(
                types.Content(
                    role="system",
                    parts=[types.Part.from_text(text=system_prompt)]
                )
            )
        
        # Process the main prompt
        if isinstance(prompt, str):
            # Single string prompt
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            )
        else:
            # List of message dictionaries
            for message in prompt:
                role = message.get("role", "user")
                content = message.get("content", "")
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=content)]
                    )
                )
        
        # Configure generation parameters
        generate_content_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            max_output_tokens=self.max_tokens,
            response_mime_type="text/plain",
        )
        
        # Return the streaming response
        return client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=generate_content_config,
        )

    def process_stream(self, stream):
        """Helper method to process the stream consistently across providers"""
        for chunk in stream:
            if self.provider == "openai":
                # Extract content from OpenAI format based on API used
                if hasattr(chunk, 'output_partial'):
                    # For responses API
                    content = chunk.output_partial
                elif hasattr(chunk, 'choices') and chunk.choices:
                    # For chat.completions API
                    if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                        content = chunk.choices[0].delta.content
                    else:
                        content = ""
                elif hasattr(chunk, 'text'):
                    # For completions API
                    content = chunk.text
                else:
                    content = ""
                    
            elif self.provider == "google":
                # Extract content from Google format
                if hasattr(chunk, 'text'):
                    content = chunk.text
                elif hasattr(chunk, 'candidates') and chunk.candidates:
                    candidate = chunk.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        content = candidate.content.parts[0].text if candidate.content.parts else ""
                    else:
                        content = ""
                else:
                    content = ""
            
            # Yield only if there's content
            if content:
                yield content


class SchemaManager:
    """
    Manages Neo4j database schema extraction and representation.
    
    This class is responsible for querying the Neo4j database to extract schema
    information, including node labels, relationship types, properties, and valid patterns.
    It also generates natural language descriptions of the schema for use in LLM prompts.
    """
    
    def __init__(self, graph: Graph, cache_duration: int = 3600):
        """
        Initialize the SchemaManager.
        
        Args:
            graph: A py2neo Graph instance connected to the Neo4j database
            cache_duration: How long to cache the schema before refreshing (in seconds)
        """
        self.graph = graph
        self.cache_duration = cache_duration
        self.schema = None
        self.last_updated = 0
        self.refresh_schema()
    
    def refresh_schema(self) -> Dict:
        """
        Extract the current schema from the Neo4j database using explicit schema introspection.
        
        Returns:
            Dict: The complete schema information
        """
        # Extract node labels using db.labels()
        label_query = """
        CALL db.labels() YIELD label
        RETURN label
        """
        
        try:
            label_result = self.graph.run(label_query).data()
            node_labels = [record["label"] for record in label_result if record["label"] is not None]
        except Exception as e:
            print(f"Error retrieving node labels: {e}")
            # Fallback to traditional method
            node_labels = []
            node_query = """
            CALL db.schema.nodeTypeProperties()
            YIELD nodeType
            RETURN DISTINCT nodeType
            """
            try:
                node_type_result = self.graph.run(node_query).data()
                for record in node_type_result:
                    if record["nodeType"] is not None:
                        clean_label = record["nodeType"].replace(':`', '').replace('`', '')
                        if clean_label.startswith(':'):
                            clean_label = clean_label[1:]
                        node_labels.append(clean_label)
            except Exception as inner_e:
                print(f"Error in fallback node label retrieval: {inner_e}")
        
        # Get properties for each node label
        nodes = {}
        for label in node_labels:
            prop_query = f"""
            MATCH (n:{label})
            RETURN properties(n) as props
            LIMIT 1
            """
            try:
                prop_result = self.graph.run(prop_query).data()
                if prop_result:
                    # Extract property names from the sample
                    props_dict = prop_result[0]["props"]
                    properties = []
                    for prop_name, value in props_dict.items():
                        prop_type = type(value).__name__
                        properties.append({
                            "name": prop_name,
                            "types": [prop_type],
                            "required": False  # Can't determine this from sample
                        })
                    nodes[label] = properties
                else:
                    # No instances of this label - create empty property list
                    nodes[label] = []
            except Exception as e:
                print(f"Error retrieving properties for {label}: {e}")
                nodes[label] = []
        
        # Extract relationship types using db.relationshipTypes()
        rel_query = """
        CALL db.relationshipTypes() YIELD relationshipType
        RETURN relationshipType
        """
        
        try:
            rel_result = self.graph.run(rel_query).data()
            rel_types = [record["relationshipType"] for record in rel_result if record["relationshipType"] is not None]
        except Exception as e:
            print(f"Error retrieving relationship types: {e}")
            # Fallback to traditional method
            rel_types = []
            backup_rel_query = """
            CALL db.schema.relTypeProperties()
            YIELD relType
            RETURN DISTINCT relType
            """
            try:
                backup_rel_result = self.graph.run(backup_rel_query).data()
                for record in backup_rel_result:
                    if record["relType"] is not None:
                        clean_rel = record["relType"].replace(':`', '').replace('`', '')
                        if clean_rel.startswith(':'):
                            clean_rel = clean_rel[1:]
                        rel_types.append(clean_rel)
            except Exception as inner_e:
                print(f"Error in fallback relationship type retrieval: {inner_e}")
        
        # Get properties for each relationship type
        relationships = {}
        for rel_type in rel_types:
            rel_prop_query = f"""
            MATCH ()-[r:{rel_type}]->()
            RETURN properties(r) as props
            LIMIT 1
            """
            try:
                rel_prop_result = self.graph.run(rel_prop_query).data()
                if rel_prop_result:
                    # Extract property names from the sample
                    props_dict = rel_prop_result[0]["props"]
                    properties = []
                    for prop_name, value in props_dict.items():
                        prop_type = type(value).__name__
                        properties.append({
                            "name": prop_name,
                            "types": [prop_type],
                            "required": False  # Can't determine this from sample
                        })
                    relationships[rel_type] = properties
                else:
                    # No instances of this relationship - create empty property list
                    relationships[rel_type] = []
            except Exception as e:
                print(f"Error retrieving properties for relationship {rel_type}: {e}")
                relationships[rel_type] = []
        
        # Extract connection patterns directly from the graph data
        pattern_query = """
        MATCH (a)-[r]->(b)
        WITH labels(a) as sourceLabels, type(r) as rel, labels(b) as targetLabels
        WHERE size(sourceLabels) > 0 AND size(targetLabels) > 0
        RETURN DISTINCT 
        sourceLabels[0] as source,
        rel as relationship,
        targetLabels[0] as target
        """
        
        patterns = []
        try:
            pattern_result = self.graph.run(pattern_query).data()
            for record in pattern_result:
                if (record["source"] is not None and 
                    record["relationship"] is not None and 
                    record["target"] is not None):
                    patterns.append({
                        "source": record["source"],
                        "relationship": record["relationship"],
                        "target": record["target"]
                    })
        except Exception as e:
            print(f"Error retrieving relationship patterns: {e}")
        
        # Get sample values for important properties (limited)
        sample_values = {}
        
        for node_type in nodes:
            # Skip if no properties
            if not nodes[node_type] or len(nodes[node_type]) == 0:
                continue
            
            # Get up to 3 properties
            properties = [p["name"] for p in nodes[node_type][:3] if p["name"]]
            if not properties:
                continue
            
            # Query for sample values
            sample_query = f"""
            MATCH (n:{node_type})
            WITH n LIMIT 20
            RETURN {", ".join([f"collect(distinct n.{prop})[..5] as {prop}" for prop in properties])}
            """
            
            try:
                sample_result = self.graph.run(sample_query).data()
                if sample_result:
                    sample_values[node_type] = sample_result[0]
            except Exception as e:
                # If sampling fails, continue without samples
                print(f"Error sampling values for {node_type}: {e}")
        
        # Compile the complete schema
        self.schema = {
            "nodes": nodes,
            "relationships": relationships,
            "patterns": patterns,
            "samples": sample_values
        }
        
        self.last_updated = time.time()
        return self.schema
    
    def get_schema(self, force_refresh: bool = False) -> Dict:
        """
        Get the database schema, refreshing if necessary.
        
        Args:
            force_refresh: Whether to force a schema refresh
            
        Returns:
            Dict: The complete schema information
        """
        current_time = time.time()
        if force_refresh or (current_time - self.last_updated) > self.cache_duration:
            return self.refresh_schema()
        return self.schema
    
    def get_schema_description(self, detail_level: str = "high") -> str:
        """
        Generate a natural language description of the schema.
        
        Args:
            detail_level: The level of detail to include ("low", "medium", "high")
            
        Returns:
            str: A natural language description of the schema
        """
        schema = self.get_schema()
        description = []
        
        # Always include basic node and relationship info
        description.append("## Node Types")
        for node_type, properties in schema["nodes"].items():
            if detail_level == "low":
                description.append(f"- `:{node_type}`")
            else:
                prop_desc = ", ".join([f"`{p['name']}`" for p in properties if p["name"]])
                if prop_desc:
                    description.append(f"- `:{node_type}` with properties: {prop_desc}")
                else:
                    description.append(f"- `:{node_type}` (no properties)")
        
        description.append("\n## Relationship Types")
        for rel_type, properties in schema["relationships"].items():
            if detail_level == "low":
                description.append(f"- `:{rel_type}`")
            else:
                prop_desc = ", ".join([f"`{p['name']}`" for p in properties if p["name"]])
                if prop_desc:
                    description.append(f"- `:{rel_type}` with properties: {prop_desc}")
                else:
                    description.append(f"- `:{rel_type}` (no properties)")
        
        # Include connection patterns for medium and high detail
        if detail_level != "low":
            description.append("\n## Common Patterns")
            pattern_groups = {}
            for pattern in schema["patterns"]:
                rel = pattern["relationship"]
                if rel not in pattern_groups:
                    pattern_groups[rel] = []
                pattern_groups[rel].append(f"(`:{pattern['source']}`)-[:`{rel}`]->(`:{pattern['target']}`)")
            
            for rel, patterns in pattern_groups.items():
                for pattern in patterns:
                    description.append(f"- {pattern}")
        
        # Include sample values only for high detail
        if detail_level == "high" and schema["samples"]:
            description.append("\n## Sample Property Values")
            for node_type, props in schema["samples"].items():
                description.append(f"\nFor `:{node_type}` nodes:")
                for prop, values in props.items():
                    value_str = ", ".join([f'"{v}"' if isinstance(v, str) else str(v) for v in values if v is not None])
                    if value_str:
                        description.append(f"- `{prop}`: {value_str}")
        
        return "\n".join(description)
    

class CypherWorkflow:
    """
    Orchestrates the multi-step workflow for generating and executing Cypher queries.
    
    This class implements the agentic workflow with query generation, verification,
    execution, and error handling steps.
    """

    def __init__(self, graph: Graph, schema_manager: SchemaManager, 
                agent1, agent2, db_mappings_text: str = "", max_retries: int = 3,
                query_history: List = None):
        self.graph = graph
        self.schema_manager = schema_manager
        self.agent1 = agent1
        self.agent2 = agent2
        self.db_mappings_text = db_mappings_text
        self.max_retries = max_retries
        self.query_history = query_history if query_history is not None else []
        self.schema = self.schema_manager.get_schema()
        self.schema_desc = self.schema_manager.get_schema_description(detail_level="high")
        
        # Initialize last_state as a class attribute
        self.last_state = {
            "user_prompt": "",
            "current_step": "init",
            "description": "Workflow initialized but not yet started.",
            "temp_cache": {},
            "result": None
        }
    
    def process_user_request(self, user_prompt: str) -> Dict:
        """
        Process a user request through the complete workflow.
        
        Args:
            user_prompt: The user's natural language prompt
            
        Returns:
            Dict: The workflow result with query, execution result, and description
        """
        # Update last_state for the new request
        self.last_state.update({
            "user_prompt": user_prompt,
            "current_step": "validation",
            "description": "Starting validation of user prompt against schema.",
            "temp_cache": {}
        })

        # Validate prompt against schema first
        validation = self._validate_prompt_against_schema(user_prompt)
        
        # If invalid, return early with suggestions
        if not validation["valid"]:
            # Create schema mismatch result
            mismatch_result = {
                "user_prompt": user_prompt,
                "cypher_query": None,
                "result": None,
                "description": "Schema terminology mismatch detected.",
                "success": False,
                "schema_mismatch": True,
                "mismatched_terms": validation["mismatched_terms"],
                "suggestions": validation["suggestions"]
            }
            
            # Update last_state before returning
            self.last_state["current_step"] = "complete"
            self.last_state["description"] += "\nSchema terminology mismatch detected."
            self.last_state["result"] = mismatch_result
            
            return mismatch_result

        # Initialize workflow state
        workflow_state = {
            "user_prompt": user_prompt,
            "attempts": [],
            "retry_count": 0,
            "feedback_attempts": 0,
            "current_step": "generate_query",
            "temp_cache": {},
            "description": "",
            "result": None,
            "max_retries": self.max_retries
        }
        
        # Process until complete
        while workflow_state["current_step"] != "complete":
            # Execute current step and get next step
            next_step = getattr(self, f"_step_{workflow_state['current_step']}")(workflow_state)
            workflow_state["current_step"] = next_step
        
        # Update last_state with the final workflow state
        self.last_state = workflow_state

        return workflow_state["result"]
    
    def get_format_values(self, **additional_values):
        """
        Create a dictionary of values for formatting prompts.
        
        Args:
            additional_values: Any additional values to include in the formatting
            
        Returns:
            Dict with values for placeholder substitution
        """
        # Base values available for all prompt types
        format_dict = {
            "schema": self.schema,
            "schema_desc": self.schema_desc,
            "db_mappings_text": self.db_mappings_text
        }
        
        # Update with any additional values provided
        format_dict.update(additional_values)
        
        return format_dict
    
    def _validate_prompt_against_schema(self, prompt: str) -> Dict:
        """
        Uses agent2 to validate the user prompt against the available schema to detect mismatches.
        
        Args:
            prompt: The user's natural language prompt
            
        Returns:
            Dict: Validation result with 'valid' flag and any suggestions
        """
        
        # Create the validation prompt
        validation_prompt = f"""
        Carry out validation of the below user request using the accompanying guidance.

        User request: "{prompt}"
        
        Respond with a JSON object as described in the guidance.
        """
        
        try:
            # Get validation from agent2

            format_values = self.get_format_values()
            validation_response = self.agent2.get_output(
                prompt=validation_prompt,
                system_prompt=SCHEMA_VALIDATION_PROMPT.format(**format_values))

            # Find JSON pattern in the response
            json_match = re.search(r'\{[\s\S]*\}', validation_response)
            if json_match:
                json_str = json_match.group(0)
                validation_result = json.loads(json_str)
                
                # Ensure expected fields exist
                if "valid" not in validation_result:
                    validation_result["valid"] = True
                if "mismatched_terms" not in validation_result:
                    validation_result["mismatched_terms"] = []
                if "suggestions" not in validation_result:
                    validation_result["suggestions"] = {}
                    
                return validation_result
            
            # If no JSON found, return valid by default
            return {"valid": True, "mismatched_terms": [], "suggestions": {}}
            
        except Exception as e:
            print(f"Error in terminology validation: {e}")
            # In case of error, allow the process to continue
            return {"valid": True, "mismatched_terms": [], "suggestions": {}}


    def _step_generate_query(self, state: Dict) -> str:
        """
        Step 1: Generate a Cypher query from the user prompt.
        
        Args:
            state: The current workflow state
            
        Returns:
            str: The next step name
        """
        user_prompt = state["user_prompt"]
        verification_feedback = state["temp_cache"].get("verification_feedback", "")
        
        # Reset relevant temp_cache fields
        state["temp_cache"] = {
            key: value for key, value in state.get("temp_cache", {}).items() 
            if key not in ["cypher_query", "result", "error", "corrected_query", 
                        "secondary_error", "reformulated_query", "tertiary_error"]
        }
        
        # Keep verification feedback if it exists
        if verification_feedback:
            state["temp_cache"]["verification_feedback"] = verification_feedback
        
        # Format the verification feedback as needed for the prompt
        verification_feedback_section = ""
        if verification_feedback:
            verification_feedback_section = f"""
            ## Previous Attempt Feedback
            Your previous query did not match the user's intent. The verification agent noted:
            "{verification_feedback}"
            
            Please address this feedback in your new query.
            """
        
        # Generate query using the updated prompt format
        format_values = self.get_format_values(verification_feedback=verification_feedback_section)
        cypher_query = self.agent2.get_output(
            prompt=user_prompt,
            system_prompt=CYPHER_GENERATION_PROMPT.format(**format_values)
)
        
        # Clean up query (remove markdown, comments, etc.)
        cypher_query = self._clean_query(cypher_query)

        # Add to state temp cache for use in verification step
        state["temp_cache"]["cypher_query"] = cypher_query
        
        # Add to query history
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_prompt": user_prompt,
            "query": cypher_query,
            "step": "generate",
            "attempt": state["retry_count"] + 1,
            "session_id": id(state)  # Use state object ID to identify the session
        })
        
        # Describe what happened
        state["description"] += f"1. Generated Cypher query based on the prompt: '{user_prompt}'\n"
        
        return "verify_query"
    
    def _step_verify_query(self, state: Dict) -> str:
        """
        Step 2: Verify the generated query matches user intent.
        
        Args:
            state: The current workflow state
            
        Returns:
            str: The next step name
        """
        user_prompt = state["user_prompt"]
        cypher_query = state["temp_cache"]["cypher_query"]
        
        # Verify query using updated prompt format
        verification_prompt = f"""
        User request: {user_prompt}
        
        Cypher query: {cypher_query}
        
        Does this query match the user's intent?
        """

        format_values = self.get_format_values()
        verification = self.agent2.get_output(
            prompt=verification_prompt,
            system_prompt=QUERY_VERIFICATION_PROMPT.format(**format_values)
)

        # Extract verdict and explanation
        lines = [line.strip() for line in verification.strip().splitlines()]
        
        # Look for standalone YES/NO answers first
        verdict_line = next((line.upper() for line in lines if line.upper() == "YES" or line.upper() == "NO"), None)

        # If not found, look for numbered points with YES/NO (like "2. YES")
        if verdict_line is None:
            for line in lines:
                if re.search(r'\d+\.\s+(YES|NO)', line.upper()):
                    verdict_line = "YES" if "YES" in line.upper() else "NO"
                    break
        
        # Add verification result to history
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "query_ref": state["temp_cache"]["cypher_query"],
            "verification_result": verification,
            "verdict": verdict_line if verdict_line else "UNKNOWN",
            "step": "verify",
            "attempt": state["retry_count"] + 1,
            "session_id": id(state)
        })

        # Continue with retry logic
        if verdict_line is None or verdict_line == "NO":
            # Increment retry counter
            state["retry_count"] += 1
            
            # Find explanation
            explanation = ""
            collect_explanation = False
            
            for line in lines:
                if collect_explanation:
                    explanation += line + " "
                elif line.upper() == "NO":
                    collect_explanation = True
            
            # Check if we've exceeded max retries
            if state["retry_count"] >= self.max_retries:
                state["description"] += f"2. Verification failed after {state['retry_count']} attempts - moving to fallback\n"
                # Ensure we have an error message for the tertiary failure step
                state["temp_cache"]["tertiary_error"] = "Verification failed after maximum retries"
                return "handle_tertiary_failure"
                
            # Otherwise continue with retry
            if explanation:
                state["temp_cache"]["verification_feedback"] = explanation.strip()
                state["description"] += f"2. Verification failed - {explanation.strip()}\n"
            else:
                state["temp_cache"]["verification_feedback"] = "Query doesn't match user intent. That's all we can say."
                state["description"] += f"2. Verification failed without explanation - moving to fallback\n"
                # Ensure we have an error message for the tertiary failure step
                state["temp_cache"]["tertiary_error"] = "Verification failed without clear explanation"
                return "handle_tertiary_failure"  # Go to fallback when no useful feedback available
            return "generate_query"  # Go back to generation

        # Verification passed
        state["description"] += "2. Verification passed - query matches user intent\n"
        return "execute_query"
    
    def _step_execute_query(self, state: Dict) -> str:
        """
        Step 3: Execute the Cypher query.
        
        Args:
            state: The current workflow state
            
        Returns:
            str: The next step name
        """
        cypher_query = state["temp_cache"]["cypher_query"]
        
        # Execute query
        try:
            result = self.graph.run(cypher_query).data()
            
            # Success
            state["temp_cache"]["result"] = result
            state["description"] += f"3. Query executed successfully with {len(result)} results\n"
            return "get_user_feedback"
            
        except Exception as e:
            # Query failed
            error = str(e)
            state["temp_cache"]["error"] = error
            state["description"] += f"3. Query execution failed with error: {error}\n"
            return "handle_error"
    
    def _step_get_user_feedback(self, state: Dict) -> str:
        """
        Step 4a: Handle results and present to user.
        
        Args:
            state: The current workflow state
            
        Returns:
            str: The next step name
        """
        user_prompt = state["user_prompt"]
        cypher_query = state["temp_cache"]["cypher_query"]
        result = state["temp_cache"]["result"]
        
        # Store for final output
        state["result"] = {
            "user_prompt": user_prompt,
            "cypher_query": cypher_query,
            "result": result,
            "description": state["description"],
            "success": True
        }
        
        return "complete"
    
    def _step_handle_error(self, state: Dict) -> str:
        """
        Step 4b: Handle query execution error.
        
        Args:
            state: The current workflow state
            
        Returns:
            str: The next step name
        """
        user_prompt = state["user_prompt"]
        cypher_query = state["temp_cache"]["cypher_query"]
        error = state["temp_cache"]["error"]
        
        # Create error correction prompt
        error_correction_prompt = f"""
        User request: {user_prompt}
        
        Failed Cypher query: {cypher_query}
        
        Error message: {error}
        
        Please correct the query.
        """
        
        # Generate corrected query
        format_values = self.get_format_values()
        corrected_query = self.agent1.get_output(
            prompt=error_correction_prompt,
            system_prompt=ERROR_CORRECTION_PROMPT.format(**format_values)
        )
        
        # Clean up query
        corrected_query = self._clean_query(corrected_query)
        
        # Save to state
        state["temp_cache"]["corrected_query"] = corrected_query
        state["attempts"].append({"query": corrected_query, "step": "error_correction"})
        state["description"] += "4b. Generated corrected query to address error\n"
        
        # Execute corrected query
        try:
            result = self.graph.run(corrected_query).data()
            
            # Success
            state["temp_cache"]["result"] = result
            state["temp_cache"]["cypher_query"] = corrected_query  # Update the current query
            state["description"] += f"    - Corrected query executed successfully with {len(result)} results\n"
            return "get_user_feedback"
            
        except Exception as e:
            # Corrected query also failed
            state["temp_cache"]["secondary_error"] = str(e)
            state["description"] += f"    - Corrected query also failed with error: {str(e)}\n"
            return "handle_secondary_error"
    
    def _step_handle_secondary_error(self, state: Dict) -> str:
        """
        Step 4c: Handle error after first correction attempt.
        
        Args:
            state: The current workflow state
            
        Returns:
            str: The next step name
        """
        user_prompt = state["user_prompt"]
        cypher_query = state["temp_cache"]["cypher_query"]
        error = state["temp_cache"]["error"]
        corrected_query = state["temp_cache"]["corrected_query"]
        secondary_error = state["temp_cache"]["secondary_error"]
        
        # Create reformulation prompt
        reformulation_prompt = f"""
        User request: {user_prompt}
        
        Failed query 1: {cypher_query}
        Error 1: {error}
        
        Failed query 2: {corrected_query}
        Error 2: {secondary_error}
        
        Please reformulate the query completely.
        """
        
        # Generate reformulated query
        format_values = self.get_format_values()
        reformulated_query = self.agent2.get_output(
            prompt=reformulation_prompt,
            system_prompt=QUERY_REFORMULATION_PROMPT.format(**format_values)
        )
        
        # Clean up query
        reformulated_query = self._clean_query(reformulated_query)
        
        # Save to state
        state["temp_cache"]["reformulated_query"] = reformulated_query
        state["attempts"].append({"query": reformulated_query, "step": "reformulation"})
        state["description"] += "4c. Completely reformulated query with the more powerful model\n"
        
        # Execute reformulated query
        try:
            result = self.graph.run(reformulated_query).data()
            
            # Success
            state["temp_cache"]["result"] = result
            state["temp_cache"]["cypher_query"] = reformulated_query  # Update the current query
            state["description"] += f"    - Reformulated query executed successfully with {len(result)} results\n"
            return "get_user_feedback"
            
        except Exception as e:
            # Reformulated query also failed
            state["temp_cache"]["tertiary_error"] = str(e)
            state["description"] += f"    - Reformulated query also failed with error: {str(e)}\n"
            return "handle_tertiary_failure"
    
    def _step_handle_tertiary_failure(self, state: Dict) -> str:
        """
        Step 4d: Handle case where all query attempts have failed.
        
        Args:
            state: The current workflow state
            
        Returns:
            str: The next step name
        """
        user_prompt = state["user_prompt"]
        
        # Collect all attempted queries
        attempted_queries = [attempt["query"] for attempt in state["attempts"]]
        
        # Ensure tertiary_error is set (even if we got here from verification failure)
        if "tertiary_error" not in state["temp_cache"]:
            state["temp_cache"]["tertiary_error"] = "Multiple attempts failed"
        
        # Create alternative suggestion prompt
        suggestion_prompt = f"""
        Original user request: {user_prompt}
        
        Please suggest:
        1. A simplified alternative Cypher query that sticks to the schema below
        2. A succinct explanation of what information this simpler query will provide
        3. How it differs from what the user originally requested

        ## Schema
        {self.schema}
        """
        
        # Generate alternative suggestion

        format_values = self.get_format_values()
        alternative_suggestion = self.agent2.get_output(
            prompt=suggestion_prompt,
            system_prompt=ALTERNATIVE_SUGGESTION_PROMPT.format(**format_values)
)

        # Format the final result
        state["result"] = {
            "user_prompt": user_prompt,
            "cypher_query": None,
            "result": None,
            "result_text": f"Query execution failed after multiple attempts. Error: {state['temp_cache']['tertiary_error']}",
            "description": state["description"] + "4d. All query attempts failed\n",
            "alternative_suggestion": alternative_suggestion,
            "success": False
        }
        
        return "complete"
    
    def _clean_query(self, query: str) -> str:
        """
        Clean up a query by removing markdown, comments, etc.
        
        Args:
            query: The raw query from the LLM
            
        Returns:
            str: The cleaned query
        """
        # Remove markdown code blocks
        if query.startswith("```") and query.endswith("```"):
            query = query[query.find("\n")+1:query.rfind("```")]
        elif query.startswith("```"):
            query = query[query.find("\n")+1:]
            
        # Remove leading/trailing whitespace
        query = query.strip()
        
        # Remove any "cypher" or "neo4j" language tags
        if query.lower().startswith("cypher"):
            query = query[6:].strip()
        
        return query
    
    def _format_results(self, results: List[Dict]) -> str:
        """
        Format query results for human-readable output.
        
        Args:
            results: The query results
            
        Returns:
            str: Formatted results
        """
        if not results:
            return "No results found."
        
        # Determine keys from first result
        keys = list(results[0].keys())
        
        # Format as a simple table
        output = []
        
        # Add header
        output.append("| " + " | ".join(keys) + " |")
        output.append("| " + " | ".join(["---" for _ in keys]) + " |")
        
        # Add rows
        for row in results:
            values = []
            for key in keys:
                value = row.get(key, "")
                if isinstance(value, (dict, list)):
                    value = str(value)
                values.append(str(value))
            output.append("| " + " | ".join(values) + " |")
        
        return "\n".join(output)


class CypherAgentSystem:
    """
    Main class that integrates all components of the Cypher Agent System.
    
    This class provides a simple interface for initializing and using the system,
    coordinating between the SchemaManager and CypherWorkflow components.
    """
    
    def __init__(self, uri: str, username: str, password: str,
                agent1_config: Dict = None, agent2_config: Dict = None,
                max_retries: int = 2):
        """
        Initialize the CypherAgentSystem.
        
        Args:
            uri: Neo4j database URI
            username: Neo4j username
            password: Neo4j password
            agent1_config: Configuration for Agent 1 (query generator)
            agent2_config: Configuration for Agent 2 (verifier)
            max_retries: Maximum number of retry attempts
        """
        # Default configurations
        self.agent1_config = agent1_config or {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.2,
        }
        
        self.agent2_config = agent2_config or {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.1,
        }
        
        # Connect to Neo4j
        self.graph = Graph(uri, auth=(username, password))
        
        # Initialize components
        self.schema_manager = SchemaManager(self.graph)
        
        # Initialize LLM agents
        self.agent1 = LLMWrapper(**self.agent1_config)
        self.agent2 = LLMWrapper(**self.agent2_config)

        # Initialize database sources
        self.db_mappings_text = self.initialize_database_sources()

        # Initialize query history storage
        self.query_history = []
        
        # Initialize workflow
        self.workflow = CypherWorkflow(
            graph=self.graph,
            schema_manager=self.schema_manager,
            agent1=self.agent1,
            agent2=self.agent2,
            db_mappings_text=self.db_mappings_text,
            max_retries=max_retries,
            query_history=self.query_history
        )

        # --- PORTIA INITIALIZATION ---
        self.portia_available = PORTIA_AVAILABLE
        self.portia = None
        if self.portia_available:
            try:
                self.portia = Portia(config=portia_config)
                print("Portia client created successfully.")
            except Exception as e:
                print(f"Error creating Portia client: {e}")
                self.portia_available = False # Mark as unavailable if init fails
        # --- END PORTIA INITIALIZATION ---

    def initialize_database_sources(self):
        """
        Query for sample nodes, identify their ID formats, and determine source databases.
        Returns a list of recognized databases for use in prompts.
        """
        recognized_sources = {}
        schema = self.schema_manager.get_schema()
        
        # Step 1: Collect ID properties and examples from all node types at once
        all_id_examples = {}
        
        for node_label, properties in schema["nodes"].items():
            # Find likely ID properties
            id_props = [p["name"] for p in properties if p["name"] and 
                    any(id_term in p["name"].lower() for id_term in 
                        ["id", "identifier", "accession", "code", "number", "name"])]
            
            if not id_props:
                continue
                
            # Query for sample nodes and extract ID values
            try:
                query = f"""
                MATCH (n:{node_label}) 
                RETURN {", ".join([f"n.{prop} as {prop}" for prop in id_props])}
                LIMIT 5
                """
                
                result = self.graph.run(query).data()
                
                if not result:
                    continue
                    
                # Collect ID values for this node type
                node_id_examples = {}
                for prop in id_props:
                    values = [row[prop] for row in result if prop in row and row[prop]]
                    if values:
                        node_id_examples[prop] = values
                
                if node_id_examples:
                    all_id_examples[node_label] = node_id_examples
                    
            except Exception as e:
                print(f"Error querying sample IDs for {node_label}: {e}")
        
        # Step 2: Send all examples in a single request to identify databases
        if all_id_examples:
            # Create a single comprehensive prompt with all ID examples
            db_prompt = DATABASE_SOURCE_PROMPT
            
            # Add all examples to the prompt
            db_prompt += json.dumps(all_id_examples, indent=2)
            
            # Get database mapping in a single API call
            db_response = self.agent2.get_output(
                prompt=db_prompt,
                system_prompt="You are an expert in biological, chemical, and medical databases and their identifier formats."
            )
            
            # Step 3: Parse the JSON response
            try:
                # Extract JSON from response if needed
                json_start = db_response.find("{")
                json_end = db_response.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = db_response[json_start:json_end]
                    db_mappings = json.loads(json_str)
                    
                    # Process mappings
                    for node_type, props in db_mappings.items():
                        for prop, db in props.items():
                            if db and db.lower() != "unknown":
                                if db not in recognized_sources:
                                    recognized_sources[db] = []
                                recognized_sources[db].append({"node_type": node_type, "property": prop})
            except Exception as e:
                print(f"Error parsing database mappings: {e}")
        
        # Step 4: Generate formatted mapping string for prompts
        db_mappings_text = "\n## Recognized Database Sources\n"
        
        if recognized_sources:
            for db_name, mappings in recognized_sources.items():
                db_mappings_text += f"- {db_name}: Used in "
                node_props = [f"{m['node_type']}.{m['property']}" for m in mappings]
                db_mappings_text += ", ".join([f"`{np}`" for np in node_props])
                db_mappings_text += "\n"
        else:
            db_mappings_text += """No specific database sources identified. Use generic ID properties."""
        
        return db_mappings_text

    def process_user_request(self, user_prompt: str, visualize=True) -> Dict:
        """
        Process a user request and return the results, with optional visualization.
        
        Args:
            user_prompt: The user's natural language prompt
            visualize: Whether to visualize successful query results
            
        Returns:
            Dict: The result including query, execution result, description, and simulated Portia plan
        """
        print(f"\nProcessing request: '{user_prompt}'")
        
        # --- DEFINE SIMULATED PORTIA PLAN STRUCTURE ---
        # This plan structure mirrors the steps in the CypherWorkflow
        simulated_plan_steps = [
            {"name": "Validate Prompt Terminology", "description": "Check if user prompt terms match DB schema", "status": PlanRunState.NOT_STARTED},
            {"name": "Generate Initial Cypher Query", "description": "Translate user prompt to initial Cypher", "status": PlanRunState.NOT_STARTED},
            {"name": "Verify Query Intent", "description": "Check if generated Cypher matches user intent", "status": PlanRunState.NOT_STARTED},
            {"name": "Execute Cypher Query", "description": "Run the verified/corrected query against Neo4j", "status": PlanRunState.NOT_STARTED},
            {"name": "Correct Query (if needed)", "description": "Attempt to fix query if execution failed", "status": PlanRunState.NOT_STARTED},
            {"name": "Reformulate Query (if needed)", "description": "Attempt full reformulation if correction failed", "status": PlanRunState.NOT_STARTED},
            {"name": "Generate Final Response", "description": "Create natural language response or suggest alternative", "status": PlanRunState.NOT_STARTED}
        ]
        # --- END PLAN DEFINITION ---

        # Process the request through the actual workflow
        result = self.workflow.process_user_request(user_prompt)
        self.last_workflow_state = self.workflow.last_state # Keep track of internal state if needed
        
        # --- SIMULATE PORTIA PLAN STATUS based on workflow result ---
        if result.get('schema_mismatch', False):
            simulated_plan_steps[0]['status'] = PlanRunState.FAILED # Validation failed
            # Other steps remain NOT_STARTED or are effectively skipped
        else:
            simulated_plan_steps[0]['status'] = PlanRunState.COMPLETE # Validation succeeded
            simulated_plan_steps[1]['status'] = PlanRunState.COMPLETE # Generation happened
            
            # Verification status (approximate based on description)
            if "Verification passed" in result.get('description', ''):
                 simulated_plan_steps[2]['status'] = PlanRunState.COMPLETE
            elif "Verification failed" in result.get('description', ''):
                 simulated_plan_steps[2]['status'] = PlanRunState.FAILED # Could be COMPLETE if retries succeeded, but FAILED is safer for demo
            else:
                 simulated_plan_steps[2]['status'] = PlanRunState.COMPLETE # Assume success if no failure mentioned

            # Execution and subsequent steps
            if result.get('success', False):
                simulated_plan_steps[3]['status'] = PlanRunState.COMPLETE # Execution succeeded
                simulated_plan_steps[4]['status'] = "Skipped" # Correction not needed
                simulated_plan_steps[5]['status'] = "Skipped" # Reformulation not needed
                simulated_plan_steps[6]['status'] = PlanRunState.COMPLETE # Final response generated
            else:
                # Execution failed at some point
                simulated_plan_steps[3]['status'] = PlanRunState.FAILED # Initial execution failed
                
                # Check if correction was attempted
                if "Generated corrected query" in result.get('description', ''):
                    if "Corrected query executed successfully" in result.get('description', ''):
                         simulated_plan_steps[4]['status'] = PlanRunState.COMPLETE # Correction succeeded
                         simulated_plan_steps[3]['status'] = PlanRunState.COMPLETE # Update execution to success via correction
                         simulated_plan_steps[5]['status'] = "Skipped"
                         simulated_plan_steps[6]['status'] = PlanRunState.COMPLETE
                    elif "Corrected query also failed" in result.get('description', ''):
                         simulated_plan_steps[4]['status'] = PlanRunState.FAILED # Correction failed
                         # Check if reformulation was attempted
                         if "Completely reformulated query" in result.get('description', ''):
                             if "Reformulated query executed successfully" in result.get('description', ''):
                                 simulated_plan_steps[5]['status'] = PlanRunState.COMPLETE # Reformulation succeeded
                                 simulated_plan_steps[3]['status'] = PlanRunState.COMPLETE # Update execution to success via reformulation
                                 simulated_plan_steps[6]['status'] = PlanRunState.COMPLETE
                             elif "Reformulated query also failed" in result.get('description', ''):
                                 simulated_plan_steps[5]['status'] = PlanRunState.FAILED # Reformulation failed
                                 simulated_plan_steps[6]['status'] = PlanRunState.COMPLETE # Alternative suggested
                             else: # Reformulation attempted but status unclear? Mark as failed for safety
                                 simulated_plan_steps[5]['status'] = PlanRunState.FAILED
                                 simulated_plan_steps[6]['status'] = PlanRunState.COMPLETE # Alternative suggested
                         else: # Correction failed, reformulation not mentioned
                             simulated_plan_steps[5]['status'] = "Skipped" # Or Failed? Skipped seems better if not mentioned
                             simulated_plan_steps[6]['status'] = PlanRunState.COMPLETE # Alternative suggested
                    else: # Correction attempted but outcome unclear
                         simulated_plan_steps[4]['status'] = PlanRunState.FAILED # Assume failed
                         simulated_plan_steps[5]['status'] = "Skipped"
                         simulated_plan_steps[6]['status'] = PlanRunState.COMPLETE # Alternative suggested
                else: # Initial execution failed, correction not mentioned
                     simulated_plan_steps[4]['status'] = "Skipped"
                     simulated_plan_steps[5]['status'] = "Skipped"
                     simulated_plan_steps[6]['status'] = PlanRunState.COMPLETE # Alternative suggested

        # Add the simulated plan to the result dictionary
        result['portiaVisualization'] = {
            "plan": {
                "name": "Cypher Query Generation (Simulated Portia Plan)",
                "description": f"Process user request: {user_prompt}"
            },
            "steps": simulated_plan_steps
        }
        # --- END SIMULATION LOGIC ---

        # Handle schema mismatch case (printing logic remains the same)
        if result.get('schema_mismatch', False):
            print(f"I noticed your query '{user_prompt}' contains terminology that doesn't match the database schema.")
            
            # Format suggestions for each mismatched term
            if result.get('suggestions'):
                print("\nHere are some possible alternatives:")
                for term, alternatives in result.get('suggestions', {}).items():
                    print(f"• Instead of '{term}', try using: {', '.join(alternatives)}")
            
            print("\nI suggest rephrasing your query using these alternatives.")
            return result
        
        # Format and display results in a more user-friendly way
        if result.get('success', False):
            query_results = result.get('result', [])
            
            # For successful queries with results
            if query_results:
                # Generate a natural language response using the query context
                self._generate_natural_language_response(user_prompt, result['cypher_query'], query_results)
                
                # Show the query for reference
                print(f"\nQuery: {result['cypher_query']}")
                
                # Visualize if requested and we have results
                if visualize and len(query_results) > 0:
                    try:
                        # Convert results to Cytoscape format
                        cytoscape_data = self._convert_to_cytoscape_format(query_results, result['cypher_query'])
                        
                        # Create visualization HTML
                        html_content = self._create_visualization_html(cytoscape_data)
                        
                        # Display visualization
                        self._display_visualization(html_content)
                    except Exception as e:
                        print(f"\nVisualization could not be generated: {e}")
                        print("Continuing with text results only.")
                        
            # For successful queries with no results
            else:
                print(f"I couldn't find any results matching your request '{user_prompt}'.")
                print(f"\nQuery executed: {result['cypher_query']}")
                print("\nWould you like to try a different query?")
        
        # For failed queries
        else:
            print(f"I couldn't successfully execute a query for '{user_prompt}'.")
            if result.get('cypher_query'):
                print(f"\nThe query I attempted was: {result['cypher_query']}")
            
            if result.get('alternative_suggestion'):
                print("\nI can suggest an alternative approach:")
                # Extract just the query from the alternative suggestion
                suggestion_lines = result['alternative_suggestion'].split('\n')
                for line in suggestion_lines:
                    if line.strip().startswith('MATCH') or line.strip().startswith('CALL'):
                        print(f"Try: {line.strip()}")
                        break
        
        # Add process description at the end for debugging/transparency
        if result.get('description'):
            print("\n--- Process Details ---")
            print(result.get('description'))
        
        return result
    
    def refresh_schema(self) -> None:
        """Refresh the database schema."""
        self.schema_manager.refresh_schema()
        print("Schema refreshed.")
    
    def get_schema_description(self, detail_level: str = "high") -> str:
        """
        Get a description of the database schema.
        
        Args:
            detail_level: The level of detail ("low", "medium", "high")
            
        Returns:
            str: The schema description
        """
        return self.schema_manager.get_schema_description(detail_level)
    
    def _generate_natural_language_response(self, user_prompt, cypher_query, results):
            """
            Use agent2 to generate a natural language response based on the query context and results,
            ensuring results passed to json.dumps are serializable.
            [Amended to fix JSON serialization error for LLM prompt context]

            Args:
                user_prompt: The original user query
                cypher_query: The executed Cypher query
                results: The query results (list of dicts from py2neo .data())
            """
            formatted_results = []
            # Limit the number of results to format for the prompt sample to avoid large context/cost
            max_results_for_prompt = 5

            for row in results[:max_results_for_prompt]: # Process only up to the limit for the prompt
                formatted_row = {}
                if not isinstance(row, dict): continue # Skip if row is not a dict

                # Process the original row, SERIALIZING values
                for key, value in row.items():
                    # Use the serialization helper here!
                    formatted_row[key] = serialize_property(value)

                # --- Logic to add '_formatted_entity' (can stay as is, uses already serialized values) ---
                identifier = None
                common_name = None
                identifier_source = None
                for key, value in formatted_row.items(): # Use the already serialized formatted_row
                    if not value: continue
                    if (key.lower().endswith('_id') or key.lower() == 'id' or \
                        key.lower() == 'identifier' or key.lower() == 'database id'):
                        identifier = value
                        identifier_source = key
                    elif (key.lower() == 'common_name' or key.lower() == 'common name' or \
                        key.lower() == 'name' or key.lower() == 'symbol' or \
                        key.lower() == 'title'):
                        common_name = value

                if common_name and identifier:
                    formatted_row['_formatted_entity'] = f"{common_name} ({identifier_source}: {identifier})"
                elif common_name:
                    formatted_row['_formatted_entity'] = common_name
                elif identifier:
                    formatted_row['_formatted_entity'] = f"{identifier_source}: {identifier}" if identifier_source else str(identifier)
                # --- End '_formatted_entity' logic ---

                formatted_results.append(formatted_row)

            # Create the response generation prompt - JSON dump should now work
            response_prompt = f"""
            User's original question: "{user_prompt}"

            Cypher query executed: {cypher_query}

            Query returned {len(results)} results.
            Sample of results (up to {max_results_for_prompt}):
            {json.dumps(formatted_results, indent=2)}

            Total number of results: {len(results)}

            Generate a natural language response based on these results and adhering to the system prompt.
            """

            try:
                # Get natural language response from agent2
                nl_response = self.agent2.get_output(
                    prompt=response_prompt,
                    system_prompt=NATURAL_LANGUAGE_RESPONSE_PROMPT
                )

                # Print the generated response
                print(nl_response)

                # --- Detailed results printing (Unchanged, doesn't use json.dumps) ---
                if len(results) > 0:
                    print("\nDetailed results:")
                    # (Keep the existing loop here that iterates through the original 'results'
                    # and prints formatted output directly without json.dumps)
                    for i, row in enumerate(results, 1):
                        # ... (your existing detailed print logic from lines 267-279) ...
                        # Example snippet of that logic:
                        row_parts = []
                        for key, value in row.items():
                            if value is not None:
                                # Maybe apply simple str() here for display if needed,
                                # but avoid complex objects if possible for clarity
                                row_parts.append(f"{key}: {str(value)[:100]}{'...' if len(str(value))>100 else ''}") # Truncate long values for display
                        print(f"• {'; '.join(row_parts)}")
                        if i < len(results): print("---")


            except Exception as e:
                # Fallback response generation (Unchanged)
                print(f"Error generating natural language response: {e}")
                print(f"I found {len(results)} results for your query:")
                # (Keep the existing fallback loop here)
                for row in results:
                    # ... (your existing fallback print logic from lines 281-291) ...
                    row_parts = []
                    for key, value in row.items():
                        if value is not None:
                            row_parts.append(f"{key}: {str(value)[:100]}{'...' if len(str(value))>100 else ''}")
                    print(f"• {'; '.join(row_parts)}")
    
    def _convert_to_cytoscape_format(self, query_results, cypher_query):
            """
            Convert Neo4j query results (list of dicts) to Cytoscape.js format.
            Handles paths, nodes, relationships found as values in the result dicts.
            [Amended for edge visibility and node label format]

            Args:
                query_results: The Neo4j query results (list of dicts from .data()).
                cypher_query: The original Cypher query (used for context/debugging).

            Returns:
                dict: Cytoscape.js compatible elements with nodes and edges,
                    or {'nodes': [], 'edges': []} if parsing fails.
            """

    def _convert_to_cytoscape_format(self, query_results, cypher_query):
                """
                ... (docstring) ...
                """
                # --- DEBUGGING STEP ---
                print(f"Debug: Received {len(query_results)} rows for visualization. First row keys (if any): {list(query_results[0].keys()) if query_results else 'N/A'}")
                # --- END DEBUGGING STEP ---

                nodes = {}
                edges = {}
                # ... (rest of the function) ...

                nodes = {}
                edges = {}

                # --- Helper Function to add Node [AMENDED LABEL LOGIC] ---
                def add_node_if_not_exists(node):
                    if not isinstance(node, Node): return None
                    node_id_str = str(node.identity)
                    if node_id_str not in nodes:
                        properties = {}
                        for k, v in node.items():
                            if not isinstance(v, (str, int, float, bool, type(None))):
                                properties[k] = str(v)
                            else:
                                properties[k] = v

                        # --- Node Label Formatting Logic ---
                        node_type = next(iter(node.labels), "UnknownType") # Get first label or default
                        common_name = node.get('common_name', node.get('name', '')) # Get common_name or name, fallback to empty
                        # Create label in desired format, handle missing common_name
                        display_label = f"{node_type}: {common_name}" if common_name else node_type
                        # --- End Label Formatting ---

                        nodes[node_id_str] = {
                            'data': {
                                'id': node_id_str,
                                'label': display_label, # Use the formatted label
                                **properties
                            }
                        }
                        # print(f"Debug: Added Node {node_id_str} - Label: {display_label}") # Optional Debug
                    return node_id_str

                # --- Helper Function to add Edge [ADDED DEBUGGING] ---
                def add_edge_if_not_exists(rel):
                    if not isinstance(rel, Relationship): return None

                    start_node_id = add_node_if_not_exists(rel.start_node)
                    end_node_id = add_node_if_not_exists(rel.end_node)

                    if start_node_id is None or end_node_id is None:
                        # More specific debug message
                        print(f"Debug: Skipping edge {type(rel).__name__} ({rel.identity}) because start ({rel.start_node.identity}) or end ({rel.end_node.identity}) node was not added.")
                        return None

                    edge_id_str = str(rel.identity)
                    if edge_id_str not in edges:
                        properties = {}
                        for k, v in rel.items():
                            if not isinstance(v, (str, int, float, bool, type(None))):
                                properties[k] = str(v)
                            else:
                                properties[k] = v

                        edges[edge_id_str] = {
                            'data': {
                                'id': edge_id_str,
                                'source': start_node_id,
                                'target': end_node_id,
                                'label': type(rel).__name__,
                                **properties
                            }
                        }
                        # print(f"Debug: Added Edge {edge_id_str} - Type: {type(rel).__name__} Source: {start_node_id} Target: {end_node_id}") # Optional Debug
                    # else: # Optional Debug
                        # print(f"Debug: Edge {edge_id_str} already exists.")
                    return edge_id_str

                # --- Main Processing Logic (largely unchanged from previous version) ---
                if not isinstance(query_results, list):
                    print(f"Warning: Expected query_results to be a list, got {type(query_results)}. Cannot generate graph.")
                    return {'nodes': [], 'edges': []}

                for row in query_results:
                    if not isinstance(row, dict): continue
                    for value in row.values():
                        if isinstance(value, Path):
                            # print(f"Debug: Processing Path object") # Optional Debug
                            # Ensure all nodes are added first from the path
                            path_node_ids = [add_node_if_not_exists(node) for node in value.nodes]
                            # Now add edges
                            for rel in value.relationships:
                                add_edge_if_not_exists(rel)
                        elif isinstance(value, Relationship):
                            # print(f"Debug: Processing Relationship object directly") # Optional Debug
                            add_edge_if_not_exists(value)
                        elif isinstance(value, Node):
                            # print(f"Debug: Processing Node object directly") # Optional Debug
                            add_node_if_not_exists(value)
                        elif isinstance(value, list):
                            # print(f"Debug: Processing List object") # Optional Debug
                            for item in value:
                                if isinstance(item, Path):
                                    item_node_ids = [add_node_if_not_exists(node) for node in item.nodes]
                                    for rel in item.relationships: add_edge_if_not_exists(rel)
                                elif isinstance(item, Relationship): add_edge_if_not_exists(item)
                                elif isinstance(item, Node): add_node_if_not_exists(item)

                # --- Combine and return ---
                elements = {'nodes': list(nodes.values()), 'edges': list(edges.values())}
                if not elements['nodes'] and not elements['edges']:
                    print("Warning: No nodes or edges extracted for visualization. Check query results structure and parsing logic.")
                else:
                    print(f"Extracted {len(elements['nodes'])} nodes and {len(elements['edges'])} edges for visualization.")
                return elements
   
    def _create_visualization_html(self, cytoscape_data):
            """
            Create HTML with Cytoscape.js for graph visualization.
            [Amended for node label size]

            Args:
                cytoscape_data: The data in Cytoscape.js format

            Returns:
                str: HTML content with embedded visualization
            """
            cytoscape_json = json.dumps(cytoscape_data)

            # Create HTML as a raw string
            html = f'''<!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Cypher Query Results Visualization</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.23.0/cytoscape.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; }}
                #cy {{ width: 100%; height: 85vh; }}
                #controls {{ padding: 10px; background: #f5f5f5; }}
                .button {{ margin: 5px; padding: 8px 15px; background: #4CAF50; color: white; border: none; cursor: pointer; }}
                .button:hover {{ background: #45a049; }}
            </style>
        </head>
        <body>
            <div id="controls">
                <button class="button" id="fit">Fit View</button>
                <button class="button" id="grid">Grid Layout</button>
                <button class="button" id="cose">Force-Directed Layout</button>
                <button class="button" id="concentric">Concentric Layout</button>
            </div>
            <div id="cy"></div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    var cy = cytoscape({{
                        container: document.getElementById('cy'),
                        elements: {cytoscape_json},
                        style: [
                            {{ // Style for Nodes
                                selector: 'node',
                                style: {{
                                    'label': 'data(label)', // Use the label from data
                                    'background-color': '#6FB1FC',
                                    'width': 50,
                                    'height': 50,
                                    'text-valign': 'center',
                                    'text-outline-width': 2,
                                    'text-outline-color': '#FFF',
                                    'font-size': '10px', // <-- ADDED: Set node label font size
                                    'text-wrap': 'wrap', // Optional: wrap long labels
                                    'text-max-width': '80px' // Optional: constrain label width
                                }}
                            }},
                            {{ // Style for Edges
                                selector: 'edge',
                                style: {{
                                    'width': 3,
                                    'line-color': '#ccc',
                                    'target-arrow-color': '#ccc',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier',
                                    'font-size': '10px',
                                    'label': 'data(label)' // Use the label from data (relationship type)
                                }}
                            }}
                        ],
                        layout: {{
                            name: 'cose',
                            animate: false,
                            nodeDimensionsIncludeLabels: false // Adjust if needed based on label wrapping
                        }}
                    }});

                    // Event listeners for layout buttons (unchanged)
                    document.getElementById('fit').addEventListener('click', function() {{ cy.fit(); }});
                    document.getElementById('grid').addEventListener('click', function() {{ cy.layout({{name: 'grid', animate: true}}).run(); }}); // Changed animate to true for grid
                    document.getElementById('cose').addEventListener('click', function() {{ cy.layout({{name: 'cose', animate: true}}).run(); }}); // Changed animate to true for cose
                    document.getElementById('concentric').addEventListener('click', function() {{ cy.layout({{name: 'concentric', animate: true}}).run(); }}); // Changed animate to true for concentric
                }});
            </script>
        </body>
        </html>'''

            return html

    def _display_visualization(self, html_content, output_path="graph_vis/cypher_results_viz.html"):
        """
        Save the visualization HTML to a file and open it in a browser.
        
        Args:quit
            html_content: The HTML content to save
            output_path: Path where to save the HTML file
        """
        # Save HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Open in default browser
        import webbrowser
        webbrowser.open('file://' + os.path.abspath(output_path))
        
        print(f"\nVisualization saved to {output_path} and opened in your browser.")

    # Add methods to CypherAgentSystem to retrieve history
    def get_query_history(self, session_id=None, limit=None):
        """
        Get the complete history of queries and verifications.
        
        Args:
            session_id: Optional filter for a specific session
            limit: Optional limit on number of entries returned
            
        Returns:
            List of query history entries
        """
        if session_id:
            history = [entry for entry in self.query_history if entry.get("session_id") == session_id]
        else:
            history = self.query_history.copy()
        
        if limit and len(history) > limit:
            return history[-limit:]
        return history

    def get_latest_session_history(self):
        """
        Get the history for the most recent query session.
        
        Returns:
            List of query history entries for the most recent session
        """
        if not self.query_history:
            return []
            
        # Get the most recent session ID
        latest_session_id = self.query_history[-1].get("session_id")
        if latest_session_id:
            return [entry for entry in self.query_history if entry.get("session_id") == latest_session_id]
        return []

    def get_verification_feedback_history(self, session_id=None):
        """
        Get a history of verification feedback for debugging purposes.
        
        Args:
            session_id: Optional filter for a specific session
            
        Returns:
            List of dictionaries containing queries and their verification feedback
        """
        history = self.get_query_history(session_id)
        
        # Extract verify steps and pair them with the preceding generate steps
        verify_history = []
        for i, entry in enumerate(history):
            if entry.get("step") == "verify":
                # Find the corresponding generate step
                generate_entry = None
                for j in range(i-1, -1, -1):
                    if history[j].get("step") == "generate" and history[j].get("attempt") == entry.get("attempt"):
                        generate_entry = history[j]
                        break
                        
                if generate_entry:
                    verify_history.append({
                        "user_prompt": generate_entry.get("user_prompt"),
                        "query": generate_entry.get("query"),
                        "verification_result": entry.get("verification_result"),
                        "verdict": entry.get("verdict"),
                        "attempt": entry.get("attempt")
                    })
                    
        return verify_history


# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = CypherAgentSystem(
        uri="neo4j://localhost:7688",
        username="neo4j",
        password="password"
    )

    # --- Example Usage with Simulated Portia Output ---
    queries_to_run = [
         "Give me up to 5 side effects associated with compounds that treat pain",
         "Return 10 genes or proteins that are associated with both cancer and Alzheimer's",
         "Return up to 5 paths of length 3 starting at NFKB",
         "Find drugs related to 'fluoxetine' and their mechanisms" # Example that might involve validation/correction
    ]

    for user_query in queries_to_run:
        result = system.process_user_request(user_query)

        # --- Print Simulated Portia Plan ---
        if 'portiaVisualization' in result:
            plan_viz = result['portiaVisualization']
            print(f"\n📝 Simulated Portia Plan: {plan_viz['plan']['name']}")
            print(f"Description: {plan_viz['plan']['description']}")
            print("\n📊 Simulated Steps:")
            for i, step in enumerate(plan_viz['steps']):
                status_icon = "✓" if step['status'] == PlanRunState.COMPLETE else \
                              "✗" if step['status'] == PlanRunState.FAILED else \
                              "⏳" if step['status'] == PlanRunState.IN_PROGRESS else \
                              "?" if step['status'] == PlanRunState.NOT_STARTED else \
                              "-" # For 'Skipped' or other statuses
                print(f"{status_icon} Step {i+1}: {step['name']} - {step['status']}")
                if step['description']: print(f"   Description: {step['description']}")
            print("--- End of Simulated Plan ---")
        else:
            print("\n(No simulated Portia plan data available)")

        print("\n=========================================\n")

    # Optionally print full result dictionary for debugging one query
    # print("\nFull result dictionary for last query:")
    # print(json.dumps(result, indent=2, default=str)) # Use default=str for non-serializable items
