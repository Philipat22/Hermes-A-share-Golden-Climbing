"""Helper functions for LLM"""

import json
from pydantic import BaseModel
from src.llm.models import get_model, get_model_info
from src.utils.progress import progress
from src.graph.state import AgentState


def call_llm(
    prompt: any,
    pydantic_model: type[BaseModel],
    agent_name: str | None = None,
    state: AgentState | None = None,
    max_retries: int = 3,
    default_factory=None,
) -> BaseModel:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        pydantic_model: The Pydantic model class to structure the output
        agent_name: Optional name of the agent for progress updates and model config extraction
        state: Optional state object to extract agent-specific model configuration
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model
    """
    
    # Extract model configuration if state is provided and agent_name is available
    if state and agent_name:
        model_name, model_provider = get_agent_model_config(state, agent_name)
    else:
        # Use system defaults when no state or agent_name is provided
        model_name = "gpt-4.1"
        model_provider = "OPENAI"

    # Extract API keys from state if available
    api_keys = None
    if state:
        request = state.get("metadata", {}).get("request")
        if request and hasattr(request, 'api_keys'):
            api_keys = request.api_keys

    model_info = get_model_info(model_name, model_provider)
    llm = get_model(model_name, model_provider, api_keys)

    # For non-JSON support models, we can use structured output
    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            # Call the LLM
            result = llm.invoke(prompt)

            # For non-JSON support models, we need to extract and parse the JSON manually
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_response(result.content)
                if parsed_result:
                    return pydantic_model(**parsed_result)
                # Failed to extract JSON → raise to trigger retry
                raise ValueError(f"Failed to extract JSON from LLM response. Content: {result.content[:300]}...")
            else:
                return result

        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")

            if attempt == max_retries - 1:
                print(f"Error in LLM call after {max_retries} attempts: {e}")
                # Use default_factory if provided, otherwise create a basic default
                if default_factory:
                    return default_factory()
                return create_default_response(pydantic_model)

    # This should never be reached due to the retry logic above
    return create_default_response(pydantic_model)


def create_default_response(model_class: type[BaseModel]) -> BaseModel:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)


def _sanitize_json(s: str) -> str:
    """Remove control characters that break json.loads"""
    import re
    # Remove raw control chars (except \t \n \r)
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\x00]', '', s)


def _try_parse(s: str) -> dict | None:
    """Try to parse a string as JSON, returns None if fails"""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _repair_truncated_json(s: str) -> str | None:
    """
    Attempt to repair truncated JSON by closing unclosed braces/brackets/quotes.
    Returns repaired string or None if repair is impossible (no JSON found).
    """
    # Find the first { and work from there
    start = s.find("{")
    if start == -1:
        return None
    candidate = s[start:]
    
    # If the content is empty after {, nothing to do
    if len(candidate) < 2:
        return None
    
    # Count unclosed structures
    open_braces = candidate.count("{") - candidate.count("}")
    open_brackets = candidate.count("[") - candidate.count("]")
    
    # Check for unclosed string (odd number of double quotes after removing escaped ones)
    # Simple check: if the last quote-like char is a quote that starts a string
    in_string = False
    escape = False
    for ch in candidate:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
    
    # If we're inside an unclosed string, close it
    if in_string:
        candidate += '"'
    
    # Close unclosed braces/brackets
    if open_braces > 0:
        candidate += '}' * open_braces
    if open_brackets > 0:
        candidate += ']' * open_brackets
    
    return candidate


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from model response. Handles markdown fence, plain JSON, and truncation."""
    content = _sanitize_json(content)
    
    # Step 1-4: standard extraction
    candidates = []
    
    # A) Markdown fence ```json ... ```
    idx = content.find("```json")
    if idx != -1:
        rest = content[idx + 7:]
        end = rest.find("```")
        if end != -1:
            candidates.append(rest[:end].strip())
        elif rest.strip():
            candidates.append(rest.strip())
    
    # B) Markdown fence ``` ... ``` (no language tag, only if no json fence)
    if not candidates:
        idx = content.find("```")
        if idx != -1:
            rest = content[idx + 3:]
            end = rest.find("```")
            if end != -1:
                candidates.append(rest[:end].strip())
            elif rest.strip():
                candidates.append(rest.strip())
    
    # C) First { ... last } block
    brace_start = content.find("{")
    if brace_start != -1:
        brace_end = content.rfind("}")
        if brace_end > brace_start:
            candidates.append(content[brace_start:brace_end + 1])
        else:
            # No closing brace yet (truncated) — try to repair
            candidates.append(content[brace_start:])
    
    # Try each candidate, with repair on fail
    for candidate in candidates:
        if not candidate:
            continue
        
        # Try direct parse
        result = _try_parse(candidate)
        if result:
            return result
        
        # Try repair (truncated JSON)
        repaired = _repair_truncated_json(candidate)
        if repaired:
            result = _try_parse(repaired)
            if result:
                return result
    
    # Last resort: try finding any valid JSON object embedded in the response
    brace_start = content.find("{")
    if brace_start != -1:
        for end_pos in range(len(content), brace_start, -1):
            fragment = content[brace_start:end_pos]
            result = _try_parse(fragment)
            if result:
                return result
            # Try repairing each fragment
            repaired = _repair_truncated_json(fragment)
            if repaired:
                result = _try_parse(repaired)
                if result:
                    return result
    
    return None


def get_agent_model_config(state, agent_name):
    """
    Get model configuration for a specific agent from the state.
    Falls back to global model configuration if agent-specific config is not available.
    Always returns valid model_name and model_provider values.
    """
    request = state.get("metadata", {}).get("request")
    
    if request and hasattr(request, 'get_agent_model_config'):
        # Get agent-specific model configuration
        model_name, model_provider = request.get_agent_model_config(agent_name)
        # Ensure we have valid values
        if model_name and model_provider:
            return model_name, model_provider.value if hasattr(model_provider, 'value') else str(model_provider)
    
    # Fall back to global configuration (system defaults)
    model_name = state.get("metadata", {}).get("model_name") or "gpt-4.1"
    model_provider = state.get("metadata", {}).get("model_provider") or "OPENAI"
    
    # Convert enum to string if necessary
    if hasattr(model_provider, 'value'):
        model_provider = model_provider.value
    
    return model_name, model_provider
