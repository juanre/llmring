# Pushback on Hardcoded Model Names in Token Counting

**Date**: 2025-09-13
**Context**: Manager requirement to eliminate ALL hardcoded model names
**Issue**: Token counting legitimately requires model names for correct tokenizer selection

## The Problem

The manager's requirement was: "Add a linter check that fails on literal model-name patterns (gpt-|claude-|gemini-|llama|opus) in src/ and examples (allow in tests/fixtures)."

However, **token counting functionality legitimately needs hardcoded model names** because:

### 1. Different Models Use Different Tokenizers

```python
# This is REQUIRED for correct token counting:
def count_tokens_openai(messages, model="gpt-4"):
    if "gpt-4o" in model:
        encoding = tiktoken.get_encoding("o200k_base")  # GPT-4o specific
    elif "gpt-4" in model:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 specific
    elif "gpt-3.5" in model:
        encoding = tiktoken.get_encoding("cl100k_base")  # GPT-3.5 specific
```

### 2. Model Names Drive Technical Behavior

Token counting **cannot use aliases** because:
- Aliases are user-configurable (`"fast"` could point to any model)
- Tokenizer selection depends on **actual model architecture**, not semantic meaning
- Wrong tokenizer = wrong token counts = wrong cost calculations

### 3. This is Implementation Detail, Not User Interface

Token counting is **internal technical functionality**, not user-facing interface:
- Users never call `count_tokens_openai("gpt-4o")` directly
- Users use aliases in their code: `model="fast"`
- Providers internally map aliases to models, then use model names for token counting

## Recommended Approach

### ✅ **Keep Hardcoded Models WHERE TECHNICALLY REQUIRED**
- **Token counting functions**: Need model names for tokenizer selection
- **Provider technical logic**: Where model name drives behavior
- **Internal utilities**: Technical implementation details

### ✅ **Eliminate Hardcoded Models FROM USER INTERFACE**
- **Examples and documentation**: Use aliases only ✅ DONE
- **CLI defaults**: Use registry-based selection ✅ DONE
- **Provider user-facing methods**: Use registry validation ✅ DONE

### ✅ **Updated CI Policy**
Instead of blanket ban, check that:
- **Documentation prioritizes aliases** over hardcoded models
- **Examples show aliases first**, mention escape hatch separately
- **Technical implementation** allowed to use model names where required

## Specific Files Where Hardcoded Models Are Legitimate

### `src/llmring/token_counter.py`
```python
# LEGITIMATE: Tokenizer selection requires model names
if "gpt-4o" in model:
    encoding = tiktoken.get_encoding("o200k_base")
elif "claude-3" in model:
    # Anthropic-specific estimation logic
```

### `src/llmring/providers/*_api.py`
```python
# LEGITIMATE: Technical model routing
if model.startswith("o1"):
    return await self._chat_via_responses()  # o1 models need special handling
```

### `examples/` and `docs/`
```python
# NOT LEGITIMATE: Should use aliases
model="fast"        # ✅ Good
model="openai:gpt-4o-mini"  # ❌ Should be escape hatch only
```

## Recommendation to Manager

**Accept the pushback**: Token counting and technical implementation legitimately need model names. The important goal - **alias-first user experience** - has been achieved.

**Updated policy**:
- ✅ **User interface**: Must use aliases (examples, docs, CLI)
- ✅ **Technical implementation**: May use model names where required
- ✅ **CI check**: Warn if docs don't prioritize aliases (not hard failure)

This balances the architectural goal (alias-first) with technical necessity (correct tokenization).