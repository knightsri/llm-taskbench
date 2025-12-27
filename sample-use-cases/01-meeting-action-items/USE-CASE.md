# Use Case: Meeting Notes Action Item Extraction

## Metadata
- **Name:** Meeting-Notes-Action-Item-Extraction
- **Difficulty:** Moderate
- **Primary Capability:** Extraction + Inference
- **Token Range:** 2K-8K input per transcript

## Goal

Extract actionable items from raw meeting transcripts including:
- **Owner:** Who is responsible (explicit or inferred)
- **Action:** What needs to be done
- **Deadline:** When it's due (explicit, relative, or inferred)
- **Priority:** Critical/High/Medium/Low
- **Dependencies:** What blocks this or what this blocks
- **Status:** New/In-Progress/Blocked/Done (if mentioned)

Output should be structured JSON that can be imported into project management tools.

Also capture:
- **Decisions Made:** Key decisions reached during the meeting
- **Open Questions:** Unresolved items needing follow-up

## LLM Evaluation Notes

**What this tests:**
- Inference capability - deadlines are often implied ("by end of sprint", "before the launch", "ASAP")
- Owner attribution - can be ambiguous ("someone from engineering should...", "we need to...")
- Distinguishing action items from discussion points
- Handling informal language and interruptions

**Comparison metrics:**
1. **Precision:** Did the model hallucinate tasks that weren't assigned?
2. **Recall:** Did it miss implicit commitments or buried action items?
3. **Attribution accuracy:** Correct owner assignment rate
4. **Deadline extraction:** Explicit vs inferred accuracy

**Edge cases to watch:**
- Conditional tasks ("if X happens, then we need to Y")
- Delegated tasks ("Sarah, can you ask Mike to...")
- Cancelled/superseded tasks mentioned then changed
- Multi-owner tasks

## Expected Output Schema

```json
{
  "meeting_id": "string",
  "action_items": [
    {
      "id": "AI-001",
      "action": "Description of what needs to be done",
      "owner": "Person Name",
      "owner_confidence": "explicit|inferred|unclear",
      "deadline": "2025-01-15",
      "deadline_type": "explicit|relative|inferred|none",
      "deadline_raw": "by end of week",
      "priority": "critical|high|medium|low",
      "dependencies": ["AI-002"],
      "context": "Brief quote or reference from transcript",
      "status": "new|in-progress|blocked|done"
    }
  ],
  "decisions": [
    {
      "decision": "What was decided",
      "context": "Brief background",
      "stakeholders": ["Person1", "Person2"]
    }
  ],
  "open_questions": [
    {
      "question": "Unresolved item",
      "raised_by": "Person Name",
      "assigned_to": "Person Name or null"
    }
  ]
}
```

## Sample Data Files

| File | Description | Complexity |
|------|-------------|------------|
| `data/meeting-01-standup.txt` | Daily standup, short, mostly explicit items | Easy |
| `data/meeting-02-planning.txt` | Sprint planning, longer, mixed explicit/implicit | Medium |
| `data/meeting-03-incident.txt` | Incident review, urgent items, unclear ownership | Hard |
| `data/meeting-04-strategy.txt` | Strategy session, vague timelines, conditional tasks | Hard |

## Ground Truth

Located in `ground-truth/` folder with matching filenames as JSON.
