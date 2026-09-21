# Audit remediation design

Approved basis: `docs/audit/repository-audit-2026-09-08.md` and the user's 2026-09-08 decisions: implement all milestones in order; existing indexes are disposable and may require rebuilding; incompatible collections permit metadata/original inspection only, never similarity queries or ingestion.

## Architecture

Keep the Python 3.12/FastAPI/Qdrant/Alpine architecture and isolated-LAN deployment. Persist application artifacts alongside Qdrant. Distinguish content-addressed face chips from observations scoped to source media and video timecode. Version changed face identity and explicitly require rebuilding legacy indexes rather than silently mixing identities. No live collection deletion or migration is part of development.

Enforce model compatibility at the query boundary for every queried collection and at ingestion before writes. The inspection override only enables viewing metadata and original evidence. Serialize mutable inference operations, enforce watchdogs independently from encoder progress, bound uploads before multipart parsing, and clean owned temporary resources without deleting another process's data.

Preserve score semantics during bounded batch search. Label face statistics as top-N neighbors and remove the implicit zero score floor. Extract ingestion decisions after corrected behavior is characterized. Ship explicit locked runtime artifacts, notices/inventory, validated Compose configurations and minimal browser/store integration checks.

## Verification boundaries

Substantive fixes require regression coverage and combined Python/JavaScript checks. Historical audit findings describe their original revision; keep them distinct from current verified behavior.

Do not expose operator .env values or use live case data for destructive tests. Synthetic restore/rebuild/purge rehearsals belong in isolated temporary storage. Hardware and deployment checks require explicit targets; development verification does not authorize operating on an existing case database.

## Completion criteria

All audit tasks mapped to verified changes or an explicit externally blocked verification item; all High defects have regression coverage; Python/JS/lint checks complete; Compose variants validate; synthetic restoration preserves audit/report bytes; deletion preserves unrelated observations; silent encoders are stopped within deadline plus bounded cleanup; incompatible vectors cannot be searched or ingested. Hardware and actual offline-image certification must be reported honestly if unavailable.
