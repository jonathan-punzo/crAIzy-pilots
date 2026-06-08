# BRIEFING — 2026-06-08T23:45:33+02:00

## Mission
Coordinate the development of a simple, complete, and functional autonomous driving bot (v8) for TORCS on the Corkscrew track, following a 4-phase plan.

## 🔒 My Identity
- Archetype: Project Orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: c:\Universita\Intelligenza Artificiale\torcs\crAIzy-pilots\.agents\orchestrator
- Original parent: main agent
- Original parent conversation ID: a7b7eac8-9537-424c-bf5b-a4766e35a016

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: c:\Universita\Intelligenza Artificiale\torcs\crAIzy-pilots\PROJECT.md
1. **Decompose**: Broke down the project into 4 sequential phases corresponding to the user request.
2. **Dispatch & Execute**:
   - **Direct (iteration loop)**: Explorer → Worker → Reviewer → gate
   - **Delegate (sub-orchestrator)**: None for this task, we will orchestrate the phases directly or through subagents.
3. **On failure** (in this order):
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (last resort)
4. **Succession**: Self-succeed at 16 spawns. Write handoff.md, spawn successor.
- **Work items**:
  1. Phase 1: Base Sensory Bot (No KNN) [pending]
  2. Phase 2: Target Speed Sensoriale [pending]
  3. Phase 3: KNN as Advisor [pending]
  4. Phase 4: Safety Governor & Integration [pending]
- **Current phase**: 1
- **Current focus**: Establish PROJECT.md, plan.md, progress.md and start Phase 1.

## 🔒 Key Constraints
- Coordinate the development of v8 bot for TORCS on Corkscrew track.
- Follow the 4-phase plan in ORIGINAL_REQUEST.md.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.
- Hard veto on forensic audit failures.
- NEVER write, modify, or create source code files directly.
- NEVER run build/test commands yourself — require workers to do so.

## Current Parent
- Conversation ID: a7b7eac8-9537-424c-bf5b-a4766e35a016
- Updated: not yet

## Key Decisions Made
- Follow the 4-phase approach: Phase 1 (deterministic sensory bot, slow speed), Phase 2 (optimize target speed), Phase 3 (integrate KNN as advisor with clamp), Phase 4 (Safety Governor + ADAS).

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_phase1 | teamwork_preview_explorer | Phase 1 codebase exploration | in-progress | cfe6cc85-c611-42bf-9b4e-729eeb237866 |

## Succession Status
- Succession required: no
- Spawn count: 1 / 16
- Pending subagents: cfe6cc85-c611-42bf-9b4e-729eeb237866
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: c152c79c-d058-4c2c-a0d2-f99627cd7f91/task-36
- Safety timer: c152c79c-d058-4c2c-a0d2-f99627cd7f91/task-46

## Artifact Index
- c:\Universita\Intelligenza Artificiale\torcs\crAIzy-pilots\.agents\orchestrator\original_prompt.md — User prompt history
- c:\Universita\Intelligenza Artificiale\torcs\crAIzy-pilots\.agents\orchestrator\plan.md — Detailed step-by-step plan
- c:\Universita\Intelligenza Artificiale\torcs\crAIzy-pilots\.agents\orchestrator\progress.md — Execution progress heartbeat
