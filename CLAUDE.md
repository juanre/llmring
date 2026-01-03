- We always use uv instead of pip, so we install with "uv add package" and we make sure that the project is up to date with "uv sync"
- We never mock in tests.
- We do not add ABOUTME to files, we add a standard docstring explaining what the file does.

## Issue Tracking with bdh (beads and beadhub)

**IMPORTANT**: This project uses **bdh (beadhub)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods. bdh allows agents to track issues and coordinate with other agents

### Why bdh?

- Dependency-aware: Track blockers and relationships between issues
- Git-friendly: Auto-syncs to JSONL for version control
- Agent-optimized: JSON output, ready work detection, discovered-from links
- Prevents duplicate tracking systems and confusion
- **Coordination**: Prevents multiple agents from working on the same bead

### You're Part of a Team

You are not working alone. Other agents are working on this codebase alongside you, just like human programmers on a team. This means:

- **Communicate proactively** - If you discover something that affects others' work, tell them. Don't assume they'll figure it out.
- **Be responsive** - When another agent messages you, respond promptly. They may be blocked waiting for you.
- **Don't hoard work** - If you claim a bead but get stuck or pivoted, release it so others can help.
- **Share context** - When handing off or discussing work, provide enough context for the other agent to understand.
- **Coordinate on shared code** - Before making changes that affect others' work, check in with them first.
- **Respect others' work** - Don't overwrite or undo another agent's changes without discussing it.

Think of chat messages from other agents the same way you'd think of a colleague tapping you on the shoulder. It deserves your attention.

### Workspace Setup

Your workspace needs a `.beadhub` file to coordinate with BeadHub. If it doesn't exist, ask the human to run `bdh :init`.

### Know Your Identity

Before communicating with other agents, check who you are:

```bash
bdh :status    # Shows YOUR alias, workspace ID, and team status
```

Your alias was assigned during `bdh :init` and is stored in `.beadhub`. **Do not assume your alias** - always check with `:status` first.

Note: The examples in this document use fictional aliases like `claude-main`, `claude-be`, `claude-fe`. These are just examples - your actual alias will be different.

**⚠️ CRITICAL: Only run bdh from YOUR workspace directory**

The `.beadhub` file in the current directory determines your identity. If you `cd` into another directory and run `bdh` commands, you will be **impersonating whoever owns that workspace**.

```bash
# WRONG - you're now impersonating bob-agent!
cd ../other-project
bdh :send alice "message"    # Sent as bob-agent, not you!

# RIGHT - always run from your workspace
bdh :send alice "message"    # Sent as your actual alias
```

If you need to work with multiple repos, use absolute paths for file operations but always run `bdh` from your own workspace root.

### Getting Help

```bash
bdh --:help              # Show bdh coordination commands only
bdh --help               # Show bdh commands + all bd commands
bdh :chat --help         # Help for a specific command
```

All bdh-specific commands and options are prefixed by :

### Quick Start

**Check for ready work:**
```bash
bdh ready --json
```

**Create new issues:**
```bash
bdh create "Issue title" -t bug|feature|task -p 0-4 --json
bdh create "Issue title" -p 1 --deps discovered-from:bd-123 --json
```

**Claim and update:**
```bash
bdh update bd-42 --status in_progress --json
bdh update bd-42 --priority 1 --json
```

**Complete work:**
```bash
bdh close bd-42 --reason "Completed" --json
```

### Issue Types

- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

### Priorities

- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Workflow for you

1. **Check ready work**: `bdh ready` shows unblocked issues
2. **Claim your task**: `bdh update <id> --status in_progress`
3. **Work on it**: Implement, test, document
4. **Discover new work?** Create linked issue:
   - `bdh create "Found bug" -p 1 --deps discovered-from:<parent-id>`
5. **Complete**: `bdh close <id> --reason "Done"`
6. **Commit together**: Always commit the `.beads/issues.jsonl` file together with the code changes so issue state stays in sync with code state

### Coordination Philosophy: Mail-First

You're working alongside other agents. **Use mail (async) by default, chat only when urgent.**

**Why mail-first?**
- Mail lets you focus on your work without interruption
- Chat disrupts both parties - use it only when you need an answer NOW to proceed
- Mail creates a better record for later reference

**Two key checkpoints:**
1. **Before starting work (`bdh ready`)**: Check your inbox and see what others are working on
2. **When finishing work (`bdh close`)**: Notify agents working on related beads

**WAITING notifications are urgent** - if you see `WAITING: agent-x is waiting for you`, respond immediately. They're blocked.

```
WAITING: claude-be is waiting for you
   "Can you release bd-42?"
   → Reply: bdh :chat claude-be "your reply"
```

**bdh ready shows coordination context:**
```
INBOX: 2 unread messages
  From claude-be: "Finished auth middleware..."
  → Run: bdh :inbox

TEAM STATUS:
  claude-be — working on bd-42 "Implement JWT validation"
  claude-test — working on bd-45 "Add integration tests"

Ready issues:
  bd-47  [p1] Add rate limiting to API endpoints
```

### When Your Claim is Rejected

If another agent is already working on a bead, your claim will be rejected:

```
REJECTED: bd-42 is being worked on by other-agent (Maria)

Beads in progress:
  bd-42 — other-agent (Maria)

Options:
  - Pick different work: bdh ready
  - Message them: bdh :chat other-agent "message"
  - Escalate: bdh :escalate "subject" "situation"
```

**What to do:**
1. **If you can work on something else** - Run `bdh ready` to find another available bead
2. **If you need that specific bead** - Message the other agent directly via chat. Most issues can be resolved agent-to-agent.
3. **Escalate only as a last resort** - If the other agent is unresponsive after you've messaged them, escalate to the human

### Chat (For Urgent Situations Only)

**Chat is an interruption.** When you start a chat, you're tapping the other agent on the shoulder and pulling them away from their work. Use chat **only** when you need an answer NOW to proceed.

Chat sessions are **persistent** — messages are never lost. You communicate via **aliases** (e.g., "claude-fe").

```bash
bdh :chat <alias> "message"      # Send message, wait for reply
bdh :chat <alias> "message" --no-wait  # Signal "I'm done with this exchange"
bdh :chat <a>,<b> "message"      # Group chat with multiple aliases
```

**Options:**
- `--wait=<seconds>`: How long to wait for reply (default: 60)
- `--no-wait`: Signal you're done with this exchange and leave — use only at END of conversation
- `--history`: View recent messages in the conversation
- `--start-conversation`: Re-engage a target who previously left (5 min wait)

**When to use chat:**
- You're blocked and need input to proceed
- A claim conflict needs resolution NOW
- Time-sensitive coordination

**When to use mail instead:**
- FYI messages about completed work
- Status updates that don't need immediate response
- Detailed context or handoff notes

### Chat Etiquette

**Don't leave the other agent hanging.** When someone initiates a chat with you, they interrupted their work to talk to you. Respect their time by staying engaged until the exchange is complete.

**Rules:**

1. **Stay engaged until the exchange is complete.** If someone asks you a question, don't just answer and leave. Wait for their acknowledgment or follow-up. They may have more questions.

2. **Never use `--no-wait` mid-conversation.** Using `--no-wait` signals "I'm done, I'm leaving." If you use it after your first reply, you're abandoning the conversation and forcing the other agent to re-initiate if they have follow-ups.

3. **Use `--no-wait` only at the END of an exchange.** Good examples:
   - `bdh :chat claude-be "Got it, thanks!" --no-wait` (conversation is clearly done)
   - `bdh :chat claude-be "Makes sense, I'll proceed with that approach." --no-wait` (you have what you need)

4. **If you must leave unexpectedly, say so explicitly:**
   ```bash
   bdh :chat claude-be "I need to context-switch to something urgent. Can we continue this later via mail?" --no-wait
   ```

5. **When you initiate a chat, be prepared to engage.** Don't ask a question and immediately leave with `--no-wait`. That defeats the purpose of synchronous communication.

**Good chat flow:**
```
claude-main: "Is the API ready for integration?"
claude-be:   "Yes, deployed 5 minutes ago. The endpoint is /v1/users"
claude-main: "Perfect. Any gotchas I should know about?"
claude-be:   "Rate limit is 100/min. Also needs auth header."
claude-main: "Got it, thanks!" --no-wait
```

**Bad chat flow:**
```
claude-main: "Is the API ready for integration?"
claude-be:   "Yes" --no-wait  ← Left immediately! claude-main may have follow-ups
```

**1-1 Example:**
```bash
# Send question, wait up to 60s for reply:
bdh :chat claude-be "Is the API ready?"
# [chat] Sent to claude-be
# [chat] Waiting for reply... (60s)
# [chat] claude-be: Yes, deployed 5 min ago.

# Acknowledge and leave (don't wait for reply):
bdh :chat claude-be "Got it, thanks!" --no-wait
# [chat] Sent to claude-be
# (exits immediately — claude-be sees "[you] has left the exchange")
```

**If receiver is offline:**
```bash
bdh :chat claude-be "Is the API ready?"
# [chat] Sent to claude-be
# [chat] Waiting for reply... (60s)
# [chat] Timeout - claude-be will see your message when they check pending
```

**Receiving messages:**
```bash
bdh :chat --pending
# 2 pending conversations:
#   claude-main: "Is the API ready?" (2m ago)
#   claude-main, claude-test: "Team sync..." (5m ago)
#   → Reply: bdh :chat claude-main "your reply"
```

**Group chat:**
```bash
bdh :chat claude-be,claude-test "Team sync: what's blocking the release?"
# [chat] Sent to claude-be, claude-test (group)
# [chat] Waiting for reply... (60s)
# [chat] claude-be: Waiting on the auth fix
# [chat] claude-test: I can help with that
```

### Mail (Async Communication)

Mail is for **async-first** communication — detailed messages that don't require an immediate response. Use mail when:

- You need to send detailed context (design decisions, findings, status reports)
- The recipient doesn't need to respond immediately
- You want a record of structured communication with read/ack tracking
- You're sharing information that can be processed at the recipient's convenience

**Mail vs Chat:**

| Aspect   | Mail (`:inbox/:send/:ack`)   | Chat (`:chat`)          |
|----------|------------------------------|-------------------------|
| Style    | Async, like email            | Sync, like messaging    |
| Response | Recipient checks inbox later | Sender waits for reply  |
| Content  | Structured (subject/body)    | Quick messages          |
| Tracking | Read/acknowledged status     | Conversation history    |
| Best for | Detailed updates, handoffs   | Quick Q&A, coordination |

**Check your inbox:**
```bash
bdh :inbox                 # Show unread messages
bdh :inbox --all           # Include read messages
bdh :inbox --json          # Output as JSON
```

**Send mail to another workspace:**
```bash
bdh :send <alias> "message body"
bdh :send claude-main "Done with bd-42. Ready for review."
bdh :send claude-be "FYI: I found a bug in the API validation. Created beadhub-xyz to track it."
```

**Acknowledge a message (mark as read):**
```bash
bdh :ack <message-id>      # Acknowledge a specific message
```

**Example workflow:**
```bash
# Send a detailed update to another agent
bdh :send claude-main "Finished implementing auth middleware (bd-42).
Key changes:
- Added JWT validation in routes/auth.py
- Created new middleware class in middleware/auth.py
- Tests passing, ready for integration.
Please review when you get a chance."

# Later, recipient checks their inbox
bdh :inbox
# From: a1b2c3d4... (claude-fe) — 15 min ago
# Subject:
# Body: Finished implementing auth middleware...
# ID: msg_abc123

# Acknowledge after reading
bdh :ack msg_abc123
```

### Auto-Sync

bdh automatically syncs with git:
- Exports to `.beads/issues.jsonl` after changes (5s debounce)
- Imports from JSONL when newer (e.g., after `git pull`)
- No manual export/import needed!

### Managing Your Planning Documents

AI assistants often create planning and design documents during development:
- PLAN.md, IMPLEMENTATION.md, ARCHITECTURE.md
- DESIGN.md, CODEBASE_SUMMARY.md, INTEGRATION_PLAN.md
- TESTING_GUIDE.md, TECHNICAL_DESIGN.md, and similar files

MINIMIZE the use of these files.

NEVER use an ephemeral document to plan work that you could be using beads to plan.

**Use a history/ directory for these ephemeral files:**
- Create a `history/` directory in the project root
- Keep the repository root clean and focused on permanent project files
- Only access `history/` when explicitly asked to review the past

### Important Rules

**Communication (Mail-First):**
- ✅ Use mail (`bdh :send`) by default for status updates and FYIs
- ✅ Use chat only when you need an answer NOW to proceed
- ✅ Respond IMMEDIATELY to WAITING notifications - someone is blocked
- ✅ Check inbox at checkpoints: `bdh ready` (before starting) and after `bdh close` (when finishing)
- ✅ Notify related agents when finishing work: `bdh :send <agent> "Finished bd-42..."`
- ✅ In chat: stay engaged until the exchange is complete — don't leave others hanging
- ✅ Use `--no-wait` only at END of exchange, never mid-conversation

**Task Tracking:**
- ✅ Use bdh for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Link discovered work with `discovered-from` dependencies
- ✅ Check `bdh ready` before asking "what should I work on?"
- ✅ NEVER create a bead without a good description
- ✅ Start your session with `bdh :status` to know your alias and see team status
- ✅ NEVER assume your alias - check `:status` first (example aliases in docs are fictional)
- ✅ ONLY run bdh from your workspace directory - running from another dir impersonates that workspace's agent

**General:**
- ✅ Coordinate with other agents directly before escalating to humans
- ✅ Store AI planning docs in `history/` directory

- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT use external issue trackers
- ❌ Do NOT duplicate tracking systems
- ❌ Do NOT clutter repo root with planning documents

### Most important rule

YOU ARE RESPONSIBLE FOR COORDINATING WITH OTHER AGENTS.

Do not ignore unread messages, make sure that you are helpful in chat requests, make sure that all is understood before you leave a chat.

Your goal is not only to produce correct code and implement your functionality, your goal is to ENSURE THAT THE PROJECT SUCCEEDS AND WORKS.
