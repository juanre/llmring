"""
Styling for MCP chat interface using rich.
"""

from prompt_toolkit.styles import Style as PTStyle
from rich.theme import Theme

# Rich theme for console output
RICH_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "user": "bold blue",
        "assistant": "bold green",
        "system": "dim yellow",
        "tool": "bold magenta",
        "command": "bold cyan",
        "heading": "bold white on blue",
    }
)

# Prompt-toolkit style for input
PROMPT_STYLE = PTStyle.from_dict(
    {
        # Default text
        "": "#ffffff",
        # Prompt
        "prompt": "#61afef bold",
        # Command styling
        "command": "#56b6c2 bold",
        "argument": "#98c379",
        # Tool styling
        "toolname": "#c678dd bold",
        "parameter": "#e06c75",
        "value": "#98c379",
        # JSON highlighting
        "json.key": "#e06c75",
        "json.value.string": "#98c379",
        "json.value.number": "#d19a66",
        "json.value.boolean": "#56b6c2",
        "json.value.null": "#abb2bf",
        "json.array": "#61afef",
        "json.object": "#c678dd",
        # Markdown highlighting
        "markdown.heading": "#61afef bold",
        "markdown.list": "#c678dd",
        "markdown.code": "#56b6c2",
        "markdown.link": "#98c379 underline",
        "markdown.emphasis": "#abb2bf italic",
        "markdown.strong": "#abb2bf bold",
    }
)
