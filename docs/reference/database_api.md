# Database API

Tracks candidate programs, archive state, island membership, and auxiliary
metadata used for sampling and visualization.

---

## `DatabaseConfig`

Controls island topology, archive size, migration, and parent-selection behavior.

::: shinka.database.dbase.DatabaseConfig
    handler: python
    options:
      show_source: false

---

## `Program`

Persisted candidate program with lineage, metrics, embeddings, and metadata.

::: shinka.database.dbase.Program
    handler: python
    options:
      show_source: false

---

## `ProgramDatabase`

Main synchronous database interface.

::: shinka.database.dbase.ProgramDatabase
    handler: python
    options:
      show_source: false
      members:
        - __init__
        - add_program
        - get_best_program
        - get_archive

---

## Prompt Database Types

Prompt co-evolution uses a dedicated prompt database layer.

::: shinka.database.prompt_dbase.SystemPromptConfig
    handler: python
    options:
      show_source: false

---

::: shinka.database.prompt_dbase.SystemPromptDatabase
    handler: python
    options:
      show_source: false
      members:
        - __init__
        - add_system_prompt
