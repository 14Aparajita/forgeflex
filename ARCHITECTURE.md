# ForgeFlex architecture

```mermaid
flowchart TD
    A[Public HIPE historical power data] --> B[Timestamp alignment and quality audit]
    B --> C[Activity audit and aggregate-only inference]
    C --> D[Chronological held-out validation]
    D --> E[Retrospective submeter validation ranking]
    F[Synthetic or user-supplied job scenario] --> G[Baseline scheduler]
    H[Illustrative or user-supplied tariff] --> G
    F --> I[Tariff-aware heuristic]
    H --> I
    F --> J[Conservative deadline-preserving heuristic]
    H --> J
    G --> K[CSV/JSON metrics and decision report]
    I --> K
    J --> K
    E --> K
    K --> L[Streamlit decision dashboard]
```

Historical data diagnostics, model evaluation, retrospective sensor ranking, and scheduling simulations are separate evidence streams. The current system consolidates them for review but does not feed sensor scores into the scheduler as a coupled optimization objective.
