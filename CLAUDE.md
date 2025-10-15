# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Competition submission for NeurIPS 2025 EEG Foundation Challenge with a 10-day development timeline. Two regression tasks: Challenge 1 (30% weight) predicts response time from Contrast Change Detection task EEG; Challenge 2 (70% weight) predicts p_factor (externalizing psychopathology) from multi-task EEG. Overall score: 0.3 × NRMSE_C1 + 0.7 × NRMSE_C2.

**Phased approach**: Days 1-4 supervised baselines (safety net) → Days 5-7 movie contrastive pretraining + multitask finetuning → Days 8-9 architecture iteration (only if contrastive beats baseline) → Day 10 submission.

## Memory-First Workflow

**CRITICAL**: Always consult the memory graph before investigating anything. Memory persists across sessions and accumulates project knowledge over time.

### Workflow

```
┌─────────────────────┐
│  Question/Task      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Search Memory      │  ← mcp__memory__search_nodes
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
  Found?      Not Found?
     │           │
     ▼           ▼
 Use Info    Investigate
     │        (Read/Grep/
     │         Notebooks)
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────────┐
│  Capture Findings   │  ← create_entities, add_observations, create_relations
└─────────────────────┘
```

### When to Consult Memory

**Always start by searching memory for**:
- "How does [component] work?"
- "Where is [function/class] defined?"
- "What are the differences between Challenge 1 and Challenge 2?"
- "What optimizer/loss/scheduler does [experiment] use?"
- "Why was [decision] made?"
- "What are the known issues with [component]?"

**Examples**:
```python
# Question: "How does Challenge 1 windowing work?"
search_nodes(query="Challenge 1 windowing")  # ✅ 3 keywords

# Question: "What optimizer does movie_pretrain use?"
search_nodes(query="movie_pretrain optimizer")  # ✅ 2 keywords

# Question: "Where is NRMSE calculated?"
search_nodes(query="NRMSE calculation")  # ✅ 2 keywords

# Question: "What are the movie tasks?"
search_nodes(query="movie tasks")  # ✅ 2 keywords
```

### Investigation Flow

1. **Search memory first**: Use `search_nodes` with relevant keywords
2. **If found**: Use the information, cross-reference with code if needed
3. **If not found**: Investigate using Read/Grep/Glob/notebooks
4. **Capture findings**: ALWAYS update memory with what you learned

### Project Filtering

**The memory graph contains knowledge from multiple projects** (cerebro + ultraspeech-qt). Use relations to filter results by project.

**How it works**:
- `search_nodes()` returns **both entities AND relations**
- Relations include `part_of` connections to project entities
- Filter results by checking: `relation.to == "cerebro"` or `relation.to == "UltraSpeech-Qt"`

**Example workflow**:
```python
# Search for something that might exist in both projects
results = search_nodes("ONNX Runtime")

# Results structure:
{
  "entities": [{"name": "ONNX Runtime", ...}],
  "relations": [
    {"from": "ONNX Runtime", "to": "UltraSpeech-Qt", "relationType": "part_of"}
  ]
}

# Filter by project: Check relation.to field
# If relation.to == "UltraSpeech-Qt" → skip (different project)
# If relation.to == "cerebro" → relevant!
```

**When to filter**:
- You're working on **cerebro** (this project) and search returns ultraspeech entities
- Entity names are ambiguous (e.g., "ConfigurationService" could exist in multiple projects)
- You want to avoid cross-project confusion

**When NOT to filter**:
- Entity names are project-specific (e.g., "Challenge 1", "EEGNeX" are obviously cerebro)
- You're intentionally learning from another project's patterns
- No ambiguity in search results

**Quick check**: If an entity has `part_of → UltraSpeech-Qt` relation, it's from the ultrasound medical imaging project, not this EEG competition project.

### Search Best Practices

**The 1-3 keyword sweet spot** (empirical finding):
- ✅ **1-2 keywords**: Broad search, returns many relevant entities
  - Example: `search_nodes("Challenge 1")` → 23 entities
  - Example: `search_nodes("movie contrastive")` → 12 entities
- ✅ **2-3 keywords**: Focused search, returns targeted results
  - Example: `search_nodes("data pipeline bottleneck")` → specific optimization entities
  - Example: `search_nodes("R5 protection")` → R5 safeguards
- ❌ **5+ keywords**: Returns nothing (too restrictive)
  - `search_nodes("Challenge 1 windowing stimulus locked preprocessing")` → 0 results

**Progressive refinement pattern** (recommended workflow):
```python
# Instead of one complex query:
search_nodes("Challenge 1 windowing stimulus locked preprocessing")  # ❌ Returns nothing

# Use progressive refinement:
# Step 1: Start broad
search_nodes("Challenge 1 windowing")
# → Find: Challenge 1 entity, windowing strategies, mention of stimulus_anchor

# Step 2: Drill down on specific detail
search_nodes("stimulus anchor")
# → Find: add_aux_anchors function, create_windows_from_events usage

# Step 3: Look up exact function
search_nodes("annotate_trials_with_target")
# → Find: Function signature, parameters, pipeline position
```

**Entity name search pattern**:
Once you find an entity name in search results, search for it directly:
```python
# Step 1: Broad search
search_nodes("Challenge 1")
# → Observations mention "annotate_trials_with_target"

# Step 2: Direct entity search
search_nodes("annotate_trials_with_target")
# → Returns function entity with full details
```

**Search anti-patterns to avoid**:
- ❌ Don't try to form one perfect query with all details
- ❌ Don't assume Boolean operators work (AND, OR, NOT, quotes)
- ❌ Don't use full sentences or questions
- ✅ Do use simple 1-3 keyword phrases
- ✅ Do iterate based on results
- ✅ Do search for entity names found in observations

### What to Capture

**Always capture**:
- **Bug root causes + fixes**: "Bug in DatasetWrapper when recordings <4s: filters out after windowing instead of before"
- **Design decisions**: "Use AdamW over Adamax for supervised training because of better weight decay handling"
- **Implementation details**: "Challenge 1 uses stimulus-locked windows via create_windows_from_events with offset=0.5s"
- **Configuration meanings**: "freeze_encoder_epochs=5 means encoder weights frozen for first 5 epochs, then fine-tuned end-to-end"
- **Gotchas**: "DatasetWrapper random crop needs seed parameter for reproducibility across workers"
- **Relationships**: "movie_pretrain uses InfoNCE loss" / "MultitaskModel requires pretrained_checkpoint parameter"
- **Timeline insights**: "Days 5-7 conditional on movie contrastive beating baseline"

**Don't capture**:
- Vague descriptions: "model uses some optimizer" (too vague)
- Obvious facts: "Python uses indentation" (not project-specific)
- Temporary notes: "TODO: check this later" (not persistent knowledge)

### How to Structure Captures

**The atomic observation principle** (official MCP guidance): Each observation should answer exactly ONE question about an entity.

#### Observation Atomicity

**Test: "Can you update this independently?"**

If one detail changes but others don't, can you update just this observation?
- If YES → good atomicity ✅
- If NO → split it ❌

**Good examples (atomic facts with details)**:
```
✅ "Loss function: MSE (Mean Squared Error)"
   → Answers: "What loss?" → One fact with clarification

✅ "Optimizer: AdamW(lr=0.001, weight_decay=0.00001, betas=[0.9,0.999])"
   → Answers: "What optimizer config?" → One conceptual unit

✅ "Input shape: (batch_size, n_chans, n_times) where n_chans=129, n_times=200"
   → Answers: "What input format?" → One fact with specification

✅ "Pipeline: annotate → filter → window → label (from startkit lines 144-193)"
   → Answers: "What's the pipeline?" → One conceptual unit with source

✅ "9 excluded subjects: NDARWV769JM7, NDARME789TD2, NDARUA442ZVF, ..."
   → Answers: "Which subjects excluded?" → One fact (list is one unit)
```

**Bad examples (compound facts - should split)**:
```
❌ "Uses AdamW optimizer (lr=1e-3, weight_decay=1e-5) and CosineAnnealingLR scheduler with T_max=epochs"
   → Two facts: optimizer AND scheduler → Should be 2 observations

❌ "Data scale: startkit uses R5 mini (~30 subjects), production uses 10 releases (~1000+ subjects)"
   → Two facts: startkit scale AND production scale → Should be 2 observations

❌ "Production uses FP16 mixed precision which is faster with negligible accuracy impact"
   → Three facts: uses FP16, performance benefit, accuracy → Should be 3 observations
```

**The gray area: Conceptual units**

Some facts are naturally bundled as one unit:
- ✅ Function signature with parameters: `annotate_trials_with_target(target_field='rt_from_stimulus', epoch_length=2.0)`
- ✅ Pipeline sequence: `annotate → filter → window → label`
- ✅ Configuration set: `AdamW(lr=0.001, weight_decay=0.00001, betas=[0.9,0.999])`

**When in doubt**: Ask "If I need to update this, would I change ALL of it or just PART of it?"

#### Entity Granularity

**The 15-observation rule**: When an entity grows beyond ~15 observations, consider splitting if:
- Observations cover multiple distinct sub-topics
- You find yourself searching for "entity AND specific_detail"
- Observations could be grouped into 2-3 coherent themes

**Example: Splitting Challenge 1 entity**

Before (too broad):
```
Entity: "Challenge 1" (30 observations covering task, model, optimizer, pipeline, bottlenecks, constraints...)
```

After (focused entities):
```
Entity: "Challenge 1" (5-8 observations: task overview only)
Entity: "Challenge 1 Windowing Constraint" (3-4 observations: fixed window requirement)
Entity: "Challenge 1 Data Pipeline Bottlenecks" (4-6 observations: performance optimizations)
Entity: "Challenge1Module" (6-8 observations: training module specifics)
```

**Benefits**: Each entity searchable with 2-3 keywords, clear content, easy to maintain.

**When to merge entities**: Two entities with <3 observations each on same topic should be merged.

#### Entity Naming Conventions

**Use 2-4 word searchable phrases**:
- ✅ "Challenge 1 Windowing Constraint" (3 words, searchable as "Challenge 1", "windowing", or "constraint")
- ✅ "Movie Contrastive Pretraining Infrastructure" (4 words, multiple search angles)
- ✅ "annotate_trials_with_target" (exact function name)
- ❌ "The Windowing Constraint Issue for Challenge 1 That Was Discovered" (too long, needs 5+ keywords)
- ❌ "Various Implementation Details" (too vague, not searchable)

**Consistency matters**: Always use same terminology
- Always "Challenge 1" (not sometimes "C1" or "Task 1")
- Always "Movie Contrastive" (not "Contrastive Movie" or variations)

#### Temporal Information Lifecycle

**Avoid temporal markers without dates**:
```
❌ "Currently only contains __init__.py files"
❌ "Will be implemented in future"
❌ "Planned feature"
```

**Use lifecycle states with dates**:
```
✅ "Status as of 2025-10-13: Only contains __init__.py files (skeleton)"
✅ "IMPLEMENTED 2025-10-13: Added pickle-based caching"
✅ "DEPRECATED 2025-10-13: Old Hydra configs removed"
✅ "RESOLVED 2025-10-14: Fixed preprocessing parallelization bug"
```

**Lifecycle state markers**:
- `PLANNED:` → Feature not yet implemented
- `IN_PROGRESS:` → Currently being developed
- `IMPLEMENTED yyyy-mm-dd:` → Feature completed
- `DEPRECATED yyyy-mm-dd:` → No longer used
- `RESOLVED yyyy-mm-dd:` → Bug/issue fixed
- `Status as of yyyy-mm-dd:` → Current state snapshot

**Maintenance triggers** (check every session start):
1. Observations with "currently", "planned", "future" → Add dates or update
2. IMPLEMENTED observations >7 days old → Verify still accurate
3. PLANNED observations >7 days old → Check if completed or abandoned

**Update vs delete guidelines**:
- **Update**: When fact changes but concept remains (e.g., learning rate tuned)
- **Delete + Add**: When structure changes significantly (e.g., single model → dual heads)
- **Delete only**: When no longer relevant (e.g., "PLANNED: add caching" after "IMPLEMENTED: added caching")

#### Front-Loading Keywords

Put searchable terms **at the beginning** of observations:

```
✅ "Uses stimulus-locked windowing via create_windows_from_events with offset=0.5s"
   → Keywords "stimulus-locked", "windowing", "create_windows_from_events" appear early

❌ "With an offset of 0.5s, the system uses stimulus-locked windowing implemented via create_windows_from_events"
   → Keywords buried mid-sentence

✅ "NRMSE (Normalized Root Mean Squared Error) is the primary competition metric"
   → Acronym + full name + context early

✅ "EEGNeX(n_chans=129, n_outputs=1, n_times=200, sfreq=100) architecture for baselines"
   → Model name + key parameters first
```

### Entity Types

Use these `entityType` values for consistency:

- **project** - Overall codebase (cerebro)
- **competition** - External challenges (NeurIPS 2025 EEG Foundation Challenge)
- **task** - Challenge tasks (Challenge 1, Challenge 2)
- **model** - Neural architectures (EEGNeX, SignalJEPA, MultitaskModel)
- **architecture** - Design patterns (SlowFast, Mamba)
- **function** - Specific functions (annotate_trials_with_target, create_windows_from_events)
- **class** - Python classes (DatasetWrapper, Submission)
- **experiment** - Configured experiments (baseline_eegnex_c1, movie_pretrain, multitask_finetune)
- **training_strategy** - Training approaches (Movie Contrastive Pretraining, supervised learning)
- **dataset** - Data sources (HBN-EEG Dataset)
- **eeg_task** - Recording paradigms (contrastChangeDetection, Movie Tasks)
- **api** - External interfaces (EEGChallengeDataset)
- **library** - External dependencies (braindecode, eegdash, PyTorch, MNE-Python)
- **tool** - Development tools (uv, wandb, Hydra)
- **loss_function** - Loss functions (InfoNCE, MSE, MAE)
- **metric** - Evaluation metrics (NRMSE)
- **optimizer** - Optimizers (AdamW, Adamax)
- **scheduler** - LR schedulers (CosineAnnealingLR, CosineAnnealingWarmRestarts)
- **code_module** - Source directories (src/data, src/models, src/training, src/evaluation, notebooks, configs)
- **configuration** - Config files (configs/*)
- **reference_code** - Starter code (startkit)
- **script** - Executable scripts (local_scoring, train.py)
- **directory** - File system directories (weights directory)
- **timeline_phase** - Development phases (Days 1-4, Days 5-7, Days 8-9, Day 10)
- **person** - Project contributors (Varun)
- **framework** - Conceptual frameworks (SimCLR, JEPA)
- **infrastructure** - System components (Hydra Configuration System)

### Relation Types

**Core relation types (prefer these)**:
- **uses** - Generic usage (prefer over uses_model, uses_optimizer, uses_scheduler, uses_loss)
- **implements** / **implemented_in** - Code implements concept (bidirectional pair)
- **part_of** / **contains** - Component relationships (bidirectional pair)
- **solves** - Model/approach solves task
- **pretrains** - Pretraining strategy for model
- **requires** / **required_for** - Dependencies
- **provides** - Source provides resource
- **analyzes** - Diagnostic analyzes component
- **configures** - Config file configures entity
- **from_library** - Imported from external library

**Specialized relations (use sparingly)**:
- **adapts_from** - When heavily modifying source
- **replicates** - When exactly matching reference
- **alternative_to** - When comparing approaches
- **targets** / **evaluates** / **loads** / **loaded_by**
- **prototypes_for** / **manages** / **submits_to** / **provided_by**
- **developed_by** / **follows** / **inspired** / **based_on**
- **loss_function_for** / **calculates** / **framework_for** / **baseline_for**
- **phase_of** / **part_of_phase** / **explored_in_phase**

**Relations to avoid** (describe in observations instead):
- ❌ **should_be_adapted_to** - Too vague, use observation
- ❌ **wraps_output_of** - Too complex, use observation
- ❌ **uses_tool** / **uses_model** / **uses_optimizer** / **uses_scheduler** / **uses_loss** / **uses_dataset** - Use generic "uses" instead

#### Relation Quality Guidelines

**1. Prefer general over specific**

Entity types already indicate what's being used:
```python
✅ "Challenge1Module" → uses → "AdamW"
   (AdamW's entityType="optimizer" makes it clear)

❌ "Challenge1Module" → uses_optimizer → "AdamW"
   (Redundant specificity)
```

**2. Active voice test**

Relations should read as natural sentences:
```
✅ "Challenge1Module" uses "AdamW" = "Challenge1Module uses AdamW" ✓
✅ "movie_pretrain" pretrains "MultitaskModel" = "movie_pretrain pretrains MultitaskModel" ✓
❌ "EEGNeX" baseline_for "Challenge 1" = "EEGNeX baseline for Challenge 1" (awkward)
✅ Better: "EEGNeX" solves "Challenge 1" = "EEGNeX solves Challenge 1" ✓
```

**3. Bidirectional pairs consistency**

When creating reverse relations, use established pairs:
```
✅ implements / implemented_in
✅ part_of / contains
✅ requires / required_for
✅ loads / loaded_by

❌ Don't create: depends_on / required_for (inconsistent pair)
   Use: requires / required_for OR depends_on / depended_on_by
```

### Memory Operations (MCP Tools)

```python
# Search for existing knowledge
mcp__memory__search_nodes(query="relevant keywords")
mcp__memory__open_nodes(names=["entity_name"])

# Create new knowledge
mcp__memory__create_entities(entities=[{
    "name": "Entity Name",
    "entityType": "appropriate_type",
    "observations": [
        "Specific observation 1 with details",
        "Specific observation 2 with measurements/parameters",
        "Gotcha: Known issue and workaround"
    ]
}])

# Add to existing knowledge
mcp__memory__add_observations(observations=[{
    "entityName": "Existing Entity",
    "contents": [
        "New insight discovered during debugging",
        "Configuration change: what + why"
    ]
}])

# Connect knowledge
mcp__memory__create_relations(relations=[{
    "from": "Source Entity",
    "to": "Target Entity",
    "relationType": "appropriate_relation"
}])
```

### Mandatory Capture Scenarios

**ALWAYS update memory after**:

1. **Resolving bugs**: Capture root cause, symptoms, and fix as observations
2. **Making design decisions**: Capture what was chosen, why, and alternatives considered
3. **Discovering gotchas**: Capture the issue and workaround
4. **Implementing features**: Capture relationships between new and existing components
5. **Changing configs**: Capture what changed, why, and expected impact
6. **Learning from startkit**: Capture pipeline differences, parameter meanings, function purposes
7. **Debugging failed experiments**: Capture what failed, why, and resolution

### Why This Matters (10-Day Timeline)

- **No repeated work**: Past investigations immediately accessible
- **Faster debugging**: Known issues and fixes in memory
- **Design consistency**: Previous decisions inform current choices
- **Session continuity**: Pick up exactly where you left off
- **Living documentation**: Memory graph becomes paper/future work foundation

**Memory is not optional—it's a productivity multiplier for time-constrained work.**

### AI Time Estimation Guidelines

**Critical**: Claude operates ~50-100x faster than humans for repetitive/parallelizable tasks. Always estimate AI time, not human time.

**Examples:**
- Memory graph audit (62 entities): AI ~5-6 min, Human ~2-3 hours
- Refactor 10 config files: AI ~3-4 min, Human ~30-45 min
- Search codebase + fix 5 bugs: AI ~8-10 min, Human ~2-4 hours
- Read 20 files + summarize: AI ~2-3 min, Human ~1-2 hours

**Why AI is faster:**
- Parallel tool calls (read 10 files simultaneously)
- No context switching overhead
- Bulk memory operations (create 20 entities in one call)
- No fatigue, breaks, or distractions
- Instant recall from memory graph (no re-reading)

**Planning rule**: Estimate AI time as ~1-2% of equivalent human time for:
- File operations (read/write/edit in bulk)
- Memory CRUD operations (search, create, update in parallel)
- Code refactoring (non-creative, pattern-based)
- Documentation updates (structured content)
- Codebase exploration (grep, glob, read in parallel)

**When AI is NOT faster:**
- Creative design decisions (still need user input)
- Debugging complex logic (bottlenecked by test execution time)
- Training ML models (wall-clock time is wall-clock time)
- Waiting for external processes (compilation, downloads, long-running scripts)

**Example planning dialogue:**
```
❌ "This memory refactoring will take 2-3 hours"
✅ "This memory refactoring will take 5-6 minutes"

❌ "Updating 15 config files across the codebase will take 45 minutes"
✅ "Updating 15 config files will take 3-4 minutes with parallel edits"
```

## Documentation Lookup with Context7

When memory doesn't have the answer and you need library documentation, use **Context7 MCP tools** before diving into source code.

### When to Use Context7

**Use after checking memory for**:
- API documentation (function signatures, parameters, return types)
- Usage examples and best practices
- Configuration options for models/optimizers/schedulers
- Understanding library-specific concepts

**Priority**: Memory (fastest) → Context7 Docs (authoritative) → Source Code (detailed but slow)

### Libraries in This Project

**Core dependencies with Context7 support**:
- **braindecode** - EEG models (EEGNeX, SignalJEPA), preprocessing utilities
- **PyTorch** - torch.optim (AdamW, Adamax), torch.nn (losses, layers), DataLoader
- **MNE-Python** - Raw data handling, BIDS format, preprocessing pipelines

### Two-Step Workflow

```python
# Step 1: Resolve library name to Context7 ID
mcp__context7__resolve_library_id(libraryName="package_name")
# Returns: {library_id: "/org/project", ...}

# Step 2: Get focused documentation
mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/org/project",
    topic="specific topic or function",  # Narrows results
    tokens=3000  # Amount of context (default: 5000)
)
```

### Examples

**Example 1: Understanding EEGNeX parameters**
```python
# Question: "What parameters does EEGNeX.forward() expect?"
mcp__context7__resolve_library_id(libraryName="braindecode")
# → {library_id: "/braindecode/braindecode"}

mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/braindecode/braindecode",
    topic="EEGNeX model initialization parameters",
    tokens=3000
)

# After reading docs, capture in memory:
mcp__memory__add_observations(observations=[{
    "entityName": "EEGNeX",
    "contents": [
        "Input shape: (batch_size, n_chans, n_times) - e.g. (128, 129, 200)",
        "n_chans parameter: number of EEG channels (129 for HBN)",
        "n_times parameter: window length in samples (200 = 2s at 100Hz)",
        "sfreq parameter: sampling frequency for temporal convolutions (100 Hz)",
        "n_outputs parameter: 1 for regression (RT or p_factor prediction)",
        "Returns: (batch_size, n_outputs) tensor"
    ]
}])
```

**Example 2: Configuring AdamW optimizer**
```python
# Question: "How does weight_decay work in AdamW vs Adam?"
mcp__context7__resolve_library_id(libraryName="torch")

mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/pytorch/pytorch",
    topic="AdamW optimizer weight decay decoupled",
    tokens=2000
)

# Capture insight:
mcp__memory__add_observations(observations=[{
    "entityName": "AdamW",
    "contents": [
        "Decoupled weight decay: applied directly to weights, not gradients (unlike Adam)",
        "Better for fine-tuning: weight_decay=0.01 typical for transfer learning",
        "Supervised training uses weight_decay=0.00001, contrastive uses 0.0001"
    ]
}])
```

**Example 3: Loading BIDS data with MNE**
```python
# Question: "How to read BIDS EEG data with MNE?"
mcp__context7__resolve_library_id(libraryName="mne")

mcp__context7__get_library_docs(
    context7CompatibleLibraryID="/mne-tools/mne-python",
    topic="read_raw_bids BIDS dataset",
    tokens=2500
)
```

### Parameters

- **topic** (optional): Focus documentation on specific area
  - More specific = more relevant results
  - Examples: "EEGNeX forward pass", "AdamW weight decay", "BIDS read_raw"

- **tokens** (optional, default: 5000): Amount of documentation to retrieve
  - More tokens = more context but slower
  - Typical range: 2000-5000 for focused queries

### After Getting Documentation

**ALWAYS capture relevant findings in memory**:

```python
# Pattern: Create entities for new concepts
mcp__memory__create_entities(entities=[{
    "name": "discovered_concept",
    "entityType": "appropriate_type",
    "observations": ["Key insight from docs"]
}])

# Pattern: Add to existing entities
mcp__memory__add_observations(observations=[{
    "entityName": "existing_entity",
    "contents": [
        "Parameter meaning from docs",
        "Gotcha discovered: specific edge case behavior"
    ]
}])
```

### Tips

- **Start specific**: Use focused topics instead of broad queries
- **Verify with code**: Cross-reference documentation with actual usage in codebase
- **Adjust tokens**: Start with 3000, increase if incomplete, decrease if too verbose
- **Don't repeat**: Once captured in memory, never look up again
- **Prefer Context7 over web search**: More structured, contains code examples

### Integration with Memory Workflow

```
Question
    ↓
Search Memory (mcp__memory__search_nodes)
    ↓ not found?
Context7 Docs (mcp__context7__*)
    ↓ still unclear?
Read Source Code (Read/Grep/Glob)
    ↓ always
Capture in Memory (mcp__memory__add_observations)
```

**The goal**: Build memory graph so comprehensive that Context7 lookups become rare.

## Running Code

Cerebro is an installable package. **One-time setup** (after cloning):

```bash
cd cerebro
uv pip install -e .  # Install in editable mode
```

### CLI Commands (Lightning CLI)

```bash
# Training with Lightning CLI
uv run cerebro fit --config configs/challenge1_base.yaml
uv run cerebro fit --config configs/challenge1_mini.yaml

# With LR finder
uv run cerebro fit --config configs/challenge1_base.yaml --run_lr_finder true

# Override config values
uv run cerebro fit --config configs/challenge1_base.yaml \
    --model.init_args.lr 0.0001 \
    --data.init_args.batch_size 256

# Backward compatible (direct python)
uv run python cerebro/cli/train.py fit --config configs/challenge1_base.yaml

# Other Lightning CLI subcommands
uv run cerebro validate --config configs/challenge1_base.yaml
uv run cerebro test --config configs/challenge1_base.yaml --ckpt_path outputs/.../best.ckpt
```

### Submission Preparation Workflow

**Competition requirements**:
- Single-level zip (NO folders): `submission.zip` containing `submission.py` + model files
- Codabench environment: Python 3.10, braindecode, PyTorch, MNE-Python
- Custom packages (mamba-ssm, neuralop) NOT available → use TorchScript

**TorchScript approach** (recommended for models with heavy dependencies):

```bash
# 1. Train model (produces Lightning checkpoint)
uv run cerebro fit --config configs/challenge1_submission.yaml

# 2. Convert Lightning checkpoint to TorchScript
uv run python -m cerebro.utils.checkpoint_to_torchscript \
    --ckpt outputs/challenge1/.../best.ckpt \
    --output model_challenge_1.pt \
    --input-shape 1 129 200

# 3. Create submission.py (template in cerebro/submission/submission.py)
# Minimal submission.py:
#   def get_model_challenge_1(self):
#       return torch.jit.load(resolve_path("model_challenge_1.pt"), map_location=self.device)

# 4. Package submission (single-level zip)
cd cerebro/submission
zip -j ../../submission.zip submission.py model_challenge_1.pt model_challenge_2.pt

# 5. Test submission locally
uv run python startkit/local_scoring.py \
    --submission-zip submission.zip \
    --data-dir data \
    --output-dir outputs/local_scoring
```

**Why TorchScript?**
- Bundles model architecture + weights in single `.pt` file
- No dependency on mamba-ssm, neuralop, or cerebro package
- Codabench environment only needs PyTorch to run `torch.jit.load()`
- Smaller submission size than bundling compiled CUDA binaries

**Alternative: State dict approach** (only for braindecode models without custom dependencies):
- Extract `state_dict` from Lightning checkpoint
- Recreate architecture in submission.py using braindecode
- Load weights with `model.load_state_dict(torch.load("weights_challenge_1.pt"))`

## Critical Data Pipeline Differences

### Challenge 1 (Response Time Prediction)
**Input**: Contrast Change Detection task only
**Windowing**: Stimulus-locked windows [stim + 0.5s, stim + 2.5s] → (129, 200)
**Label**: `rt_from_stimulus` from `annotate_trials_with_target` (per-trial)
**Loss**: MSE
**Key functions**: `annotate_trials_with_target`, `add_aux_anchors`, `create_windows_from_events`, `add_extras_columns`

### Challenge 2 (P-Factor Prediction)
**Input**: Any task (CCD, RestingState, movies, surroundSupp, etc.)
**Windowing**: Fixed 4s windows, 2s stride → random 2s crops via `DatasetWrapper` → (129, 200)
**Label**: `externalizing` field from participants.tsv (per-subject, constant across windows)
**Loss**: MAE/L1
**Key difference**: DatasetWrapper provides data augmentation through random cropping

### Movie Contrastive Pretraining
**Input**: 4 movie tasks (DespicableMe, ThePresent, DiaryOfAWimpyKid, FunwithFractals)
- **Note**: All 4 movies available in ALL releases including R5 (verified empirically)
**Windowing**: 2s sliding windows (configurable stride: 1s for overlap or 2s for non-overlapping)
**Positive pairs**: (subject_i[movie_A, t], subject_j[movie_A, t]) - exact same movie and timestamp, different subjects
**Negative pairs**: Different movies (any subjects/timestamps)
**Loss**: InfoNCE with temperature scaling
**Alignment strategy**: Perfect temporal alignment (no time binning) to preserve inter-subject correlation (ISC) precision

## Architecture Patterns

### Config Composition (Hydra)
Configs are composed from base modules:
- `configs/config.yaml`: Global settings (seed, device, paths, wandb)
- `configs/data/hbn.yaml`: Dataset parameters, windowing specs, excluded subjects
- `configs/model/*.yaml`: Model architectures (eegnex, jepa, contrastive)
- `configs/training/*.yaml`: Training loops (supervised, contrastive, multitask)
- `configs/experiment/*.yaml`: Complete experiments composing above configs

Experiment configs use `defaults` to inherit and `@package _global_` to override at root level.

### Module Organization
**cerebro/data/**: Dataset classes replicating startkit pipelines exactly
- `challenge1.py`: Implements annotate→filter→window→label pipeline (Lightning DataModule)
- `challenge2.py`: Implements filter→window→wrap pipeline with DatasetWrapper
- `movies.py`: Implements positive/negative pair creation for contrastive learning

**cerebro/models/**: Model wrappers around braindecode implementations
- `challenge1.py`: Challenge1Module (Lightning Module with EEGNeX)
- `encoders.py`: Wraps braindecode.models (EEGNeX, SignalJEPA)
- `projector.py`: MLP projection head for contrastive learning
- `heads.py`: Regression heads for C1/C2
- `multitask.py`: Shared encoder + dual heads architecture

**cerebro/cli/**: Command-line interface
- `train.py`: CerebroCLI (extends Lightning CLI with logging, LR finder, batch size finder)

**cerebro/utils/**: Utilities
- `logging.py`: Rich logging setup (console + file)
- `tuning.py`: LR finder and batch size scaler wrappers
- `checkpoint_to_torchscript.py`: Convert Lightning checkpoints to TorchScript for submission (future work)

**cerebro/training/**: Training loops with different objectives (future work)
- `supervised.py`: Single-task MSE/MAE training
- `contrastive.py`: InfoNCE loss with pair sampling
- `multitask.py`: Joint C1+C2 optimization with frozen encoder phase

**cerebro/submission/**: Competition submission templates (future work)
- `submission.py`: Template Submission class with `get_model_challenge_1/2()`
- `package_submission.py`: Create competition-ready zip file
- `test_submission.py`: Local testing before Codabench upload

## Startkit Integration

The `startkit/` directory contains reference implementations that MUST be replicated exactly:
- `challenge_1.py`: Shows exact preprocessing for Challenge 1 (lines 144-193: annotate→windows→labels)
- `challenge_2.py`: Shows exact preprocessing for Challenge 2 (lines 231-259: filter→windows→wrap)
- `local_scoring.py`: Defines evaluation protocol (lines 76-114: NRMSE calculation, 107-115: overall score)
- `submission.py`: Shows required Submission class interface with `get_model_challenge_1()` and `get_model_challenge_2()`

**Critical**: Submission must be single-level zip (NO folder) containing submission.py + weights files.

## Submission Packaging

### TorchScript Conversion

**Lightning checkpoint structure**:
```python
checkpoint = torch.load("best.ckpt")
# Contains:
# - state_dict: Model weights (prefixed with 'model.')
# - optimizer_states: Not needed for inference
# - lr_schedulers: Not needed for inference
# - hparams: Not needed (architecture defined in submission.py)
```

**Conversion steps**:
```python
# 1. Load Lightning module
from cerebro.models.challenge1 import Challenge1Module
pl_module = Challenge1Module.load_from_checkpoint("best.ckpt")
model = pl_module.model  # Extract raw PyTorch model

# 2. Convert to TorchScript
model.eval()
dummy_input = torch.zeros(1, 129, 200)  # (batch, channels, time)
scripted_model = torch.jit.trace(model, dummy_input)

# 3. Optimize for inference
scripted_model = torch.jit.optimize_for_inference(scripted_model)

# 4. Save
scripted_model.save("model_challenge_1.pt")
```

**Submission.py template**:
```python
from pathlib import Path
import torch

def resolve_path(name="model_file_name"):
    """Handle Codabench mount points: /app/input/res/, /app/input/, etc."""
    if Path(f"/app/input/res/{name}").exists():
        return f"/app/input/res/{name}"
    elif Path(f"/app/input/{name}").exists():
        return f"/app/input/{name}"
    elif Path(f"{name}").exists():
        return f"{name}"
    elif Path(__file__).parent.joinpath(f"{name}").exists():
        return str(Path(__file__).parent.joinpath(f"{name}"))
    else:
        raise FileNotFoundError(f"Could not find {name}")

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        return torch.jit.load(
            resolve_path("model_challenge_1.pt"),
            map_location=self.device
        )

    def get_model_challenge_2(self):
        return torch.jit.load(
            resolve_path("model_challenge_2.pt"),
            map_location=self.device
        )
```

**Critical gotchas**:
1. **Single-level zip**: Use `zip -j` to avoid nested folders
2. **Model.eval()**: Must set eval mode before tracing (disables dropout/batchnorm)
3. **Input shape**: Trace with exact input shape model expects (batch_size can be 1)
4. **Device placement**: TorchScript preserves device, use map_location in submission.py
5. **No cerebro imports**: submission.py cannot import cerebro package (not in Codabench)

### Local Testing

**Test TorchScript model before submission**:
```python
# Load TorchScript model
model = torch.jit.load("model_challenge_1.pt")
model.eval()

# Test inference
dummy_input = torch.zeros(16, 129, 200)  # Batch of 16
with torch.inference_mode():
    output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")  # Should be (16, 1)
assert output.shape == (16, 1), "Output shape mismatch!"
```

**Test full submission workflow**:
```bash
# 1. Extract submission.zip to temp directory
unzip submission.zip -d /tmp/test_submission/

# 2. Run local scoring (replicates Codabench evaluation)
uv run python startkit/local_scoring.py \
    --submission-dir /tmp/test_submission/ \
    --data-dir data \
    --output-dir outputs/local_scoring

# 3. Check NRMSE scores
# Challenge 1 NRMSE: X.XXXX
# Challenge 2 NRMSE: X.XXXX
# Overall score: 0.3 * C1_NRMSE + 0.7 * C2_NRMSE
```

## Data Specifications

**HBN-EEG Dataset**:
- **11 releases total**: ds005505-bdf (R1) through ds005516-bdf (R11)
  - R1: ds005505-bdf
  - R2: ds005506-bdf
  - R3: ds005507-bdf
  - R4: ds005508-bdf
  - **R5: ds005509-bdf (COMPETITION VALIDATION SET - NEVER TRAIN ON THIS)**
  - R6: ds005510-bdf
  - R7: ds005511-bdf
  - R8: ds005512-bdf
  - R9: ds005514-bdf (note: ds005513 does not exist)
  - R10: ds005515-bdf
  - R11: ds005516-bdf
- **Training data**: R1-R4, R6-R11 (10 releases)
- **Competition validation data**: R5 only (held out, provides leaderboard feedback)
- 129 channels (including reference Cz), 100 Hz sampling
- 9 excluded subjects (listed in configs/data/hbn.yaml under challenge1.excluded_subjects)

**Download**: Use `EEGChallengeDataset` from eegdash library (handles caching automatically)

**participants.tsv**: Contains p_factor, age, sex, task availability flags. Note: some subjects have `n/a` for all factor scores.

**Important**: The startkit uses `mini=True` with `release="R5"` for quick demos/tutorials only. For actual training, use `mini=False` (or omit mini parameter) with `release` in ["R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "R10", "R11"].

## Competition Data Strategy

### Prototyping Phase (Days 1-9)

**Goal**: Architecture selection, hyperparameter tuning, feature engineering

**Data splits**:
- **Train**: Subset of {R1-R4, R6-R11} split at **subject level** (e.g., 80% of subjects)
- **Val**: Held-out subjects from {R1-R4, R6-R11} (e.g., 20% of subjects)
- **Test**: R5 (treated as external test, checked sparingly)

**Workflow**:
1. Split training releases at subject level to prevent data leakage
2. Use local validation for rapid iteration and hyperparameter tuning
3. Minimize R5 checks to avoid overfitting to leaderboard signal
4. Treat R5 as external "test" for architecture/hyperparameter selection

**Why subject-level splits?**
- Same subject may have multiple recordings across different tasks
- Window-level or recording-level splits risk data leakage
- Model could memorize subject-specific EEG patterns rather than task patterns

### Final Runs Phase (Day 10)

**Goal**: Maximum performance for competition leaderboard

**Data splits**:
- **Train**: ALL subjects from {R1-R4, R6-R11} (no validation split)
- **Val/Test**: R5 (final submission to competition)

**Workflow**:
1. Architecture and hyperparameters are locked (no more tuning)
2. Train on 100% of available training data to maximize model capacity
3. Submit final predictions on R5 for leaderboard evaluation
4. No local validation (all data used for training)

**Why this approach?**
- Standard competition pattern (Kaggle, NeurIPS, etc.)
- Maximizes training data when decisions are finalized
- Prevents leaderboard overfitting during development
- Local validation eliminates need for frequent R5 checks

**Critical**: R5 is a competition validation set that provides leaderboard feedback, NOT a traditional test set. Never train on R5 under any circumstances.

### Challenge1DataModule Modes

The `Challenge1DataModule` supports two modes via the `mode` parameter:

#### Dev Mode (`mode="dev"`)
**Purpose**: Development and hyperparameter tuning with local validation

**Data splits**:
- Splits training releases (R1-R4, R6-R11) into train/val/test at subject level
- `test_on_r5=false` (default): Use local test split from training releases
- `test_on_r5=true`: Use R5 as test set (matches competition evaluation)

**Configs**:
- `configs/challenge1_base.yaml` - Standard dev mode with local test
- `configs/challenge1_mini.yaml` - Fast prototyping with R1 mini dataset
- `configs/challenge1_r5test.yaml` - Dev mode with R5 test evaluation

**Usage**:
```bash
# Dev mode with local test split (fastest iteration)
uv run cerebro fit --config configs/challenge1_base.yaml

# Dev mode with R5 test (validate against competition distribution)
uv run cerebro fit --config configs/challenge1_r5test.yaml

# Mini mode for prototyping
uv run cerebro fit --config configs/challenge1_mini.yaml
```

**When to use**:
- Days 1-9: Architecture selection, hyperparameter tuning
- Use `test_on_r5=false` for fast iteration
- Use `test_on_r5=true` to validate major decisions against R5 distribution

#### Submission Mode (`mode="submission"`)
**Purpose**: Final submission with maximum training data

**Data splits**:
- Train on ALL subjects from training releases (R1-R4, R6-R11)
- No val split (uses all data for training)
- `test_on_r5=true` (required): Test on R5 for competition submission

**Config**:
- `configs/challenge1_submission.yaml` - Final submission mode

**Usage**:
```bash
# Day 10: Final submission training
uv run cerebro fit --config configs/challenge1_submission.yaml
```

**When to use**:
- Day 10 only, after architecture and hyperparameters are locked
- Maximizes training data for final submission
- No validation monitoring (save checkpoints periodically)

#### Mode Comparison Table

| Feature | Dev Mode (local test) | Dev Mode (R5 test) | Submission Mode |
|---------|----------------------|-------------------|-----------------|
| Train data | 80% of training releases | 80% of training releases | 100% of training releases |
| Val data | 10% of training releases | 10% of training releases | None |
| Test data | 10% of training releases | R5 | R5 |
| Validation monitoring | ✓ | ✓ | ✗ |
| Early stopping | ✓ | ✓ | ✗ |
| When to use | Fast iteration | Validate decisions | Final submission |
| Config | `challenge1_base.yaml` | `challenge1_r5test.yaml` | `challenge1_submission.yaml` |

#### R5 Protection

**Multiple layers of protection prevent R5 contamination**:
1. **ValueError guard**: `Challenge1DataModule.__init__()` raises error if "R5" in releases list
2. **Mode validation**: Submission mode requires `test_on_r5=True` (raises error otherwise)
3. **Separate caching**: R5 cached in `cache/challenge1/r5/` to avoid mixing with training data
4. **Validation notebook**: `notebooks/08_validate_data_quality.py` checks for train/R5 subject overlap

#### Verification

**Before training**:
```bash
# Validate data quality and R5 separation
uv run python notebooks/08_validate_data_quality.py

# Test R5 evaluation matches local_scoring.py
uv run python test_r5_evaluation.py --use-mini
```

**Expected outputs**:
- No subject overlap between training releases and R5
- R5 distribution may differ from training (expected - different subject pool)
- R5 preprocessing matches `startkit/local_scoring.py` exactly

## Development Workflow

1. **Start with mini=True**: Fast prototyping on small dataset
2. **Test local scoring early**: `uv run python scripts/evaluate.py --fast-dev-run` after every change
3. **Use notebooks for exploration**: Jupytext .py format (# %%) in notebooks/
4. **Hydra outputs**: Auto-generates timestamped directories in outputs/
5. **Wandb integration**: All experiments logged (set WANDB_API_KEY in .env)
6. **Git commits**: After each working milestone

## Priority Guidelines (10-Day Timeline)

**Must have**:
- Working supervised baselines (C1 + C2)
- Local scoring integration
- Submission packaging

**Should have**:
- Movie contrastive pretraining
- Multitask fine-tuning
- Wandb tracking

**Nice to have** (only if time permits):
- SignalJEPA baseline
- Spatial/temporal hierarchies
- Hyperparameter sweeps
- MOABB data integration

## Key Design Decisions

1. **Installable package**: Editable install (`uv pip install -e .`) for clean imports and IDE support
2. **CLI entry point**: `uv run cerebro fit` command via package entry point (backward compatible with direct python)
3. **Config-driven experiments**: Change hyperparameters via Lightning CLI, not code
4. **Aggressive caching**: EEGChallengeDataset caches downloads, preprocessed data cached via pickle
5. **Baselines first**: Supervised training is safety net before trying contrastive learning
6. **Local scoring critical**: Test with `uv run python startkit/local_scoring.py` before submitting to competition

## User Preferences (from ~/.claude/CLAUDE.md)

- Prefer editing existing files over creating new ones
- Use `uv run python ...` instead of `python ...`
- Prefer github MCP to git commands (but avoid in this workflow due to speed)
- Never add or remove packages
- Always use `uv run ...`
- Functional style over classes
- Implement prototypes as notebooks (.py with #%% for VSCode)
- Ask when faced with design choices (user has PhD)
- Verify assumptions before internalizing
- Don't pollute repo with throwaway code
- Questions are just questions - don't infer intent to act
