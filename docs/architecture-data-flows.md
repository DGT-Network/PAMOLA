# PAMOLA.CORE Data Flows and Component Interactions

**Version:** 0.1.0
**Last Updated:** 2026-03-12

## Overview

This document describes data processing flows, task execution patterns, and component interactions in PAMOLA.CORE.

## Data Processing Flow

### Anonymization Decision Flow

```mermaid
flowchart TB
    Start([Start]) --> Input[Load Input Data]
    Input --> Profile[Profile Data]
    Profile --> Risk{Assess Risk Level}

    Risk --> High[High Risk]
    Risk --> Medium[Medium Risk]
    Risk --> Low[Low Risk]

    High --> Strong[Strong Anonymization]
    Medium --> Moderate[Moderate Anonymization]
    Low --> Minimal[Minimal Anonymization]

    Strong --> Metrics[Calculate Metrics]
    Moderate --> Metrics
    Minimal --> Metrics

    Metrics --> Thresholds{Thresholds Met?}

    Thresholds --> No[No]
    Thresholds --> Yes[Yes]

    No --> Adjust[Adjust Parameters]
    Adjust --> Strong

    Yes --> Attacks[Run Attack Suite]
    Attacks --> AttackRisk{Attack Risk Acceptable?}

    AttackRisk --> No2[No]
    AttackRisk --> Yes2[Yes]

    No2 --> Adjust2[Adjust Anonymization]
    Adjust2 --> Strong

    Yes2 --> Output[Generate Output]
    Output --> Manifest[Create manifest.json]
    Manifest --> End([End])

    style High fill:#ff6b6b
    style Medium fill:#ffd93d
    style Low fill:#6bcf7f
    style No fill:#ff6b6b
    style Yes fill:#6bcf7f
    style No2 fill:#ff6b6b
    style Yes2 fill:#6bcf7f
```

### Task Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant TaskRunner
    participant Operation
    participant Progress
    participant Cache
    participant Storage

    User->>TaskRunner: Create Task with seed
    TaskRunner->>Operation: Initialize with config
    Operation->>Cache: Check for cached result
    alt Cache Hit
        Cache-->>Operation: Return cached result
    else Cache Miss
        Operation->>Operation: Validate input
        Operation->>Progress: Start progress tracking
        Operation->>Operation: Execute operation
        Operation-->>Progress: Update progress
        Operation->>Operation: Finalize result
        Operation->>Cache: Store result in cache
    end
    Operation-->>TaskRunner: Return OperationResult
    TaskRunner->>Storage: Save to manifest.json
    TaskRunner-->>User: Return result with artifacts
```

### Data Processing Pipeline

```mermaid
graph LR
    subgraph "Input"
        CSV[CSV]
        JSON[JSON]
        Excel[Excel]
        Parquet[Parquet]
    end

    subgraph "IO Adapters"
        CSVAdapter[CSV Adapter]
        JSONAdapter[JSON Adapter]
        ExcelAdapter[Excel Adapter]
        ParquetAdapter[Parquet Adapter]
    end

    subgraph "Processing"
        DataFrame[pd.DataFrame / dd.DataFrame]
    end

    subgraph "Operations"
        Op1[Operation 1]
        Op2[Operation 2]
        OpN[Operation N]
    end

    subgraph "Output"
        Output[Transformed Data]
        Metrics[Metrics JSON]
        Reports[Reports]
        Plots[Plots]
    end

    CSV --> CSVAdapter
    JSON --> JSONAdapter
    Excel --> ExcelAdapter
    Parquet --> ParquetAdapter

    CSVAdapter --> DataFrame
    JSONAdapter --> DataFrame
    ExcelAdapter --> DataFrame
    ParquetAdapter --> DataFrame

    DataFrame --> Op1
    Op1 --> Op2
    Op2 --> OpN
    OpN --> Output

    Op1 --> Metrics
    Op2 --> Metrics
    OpN --> Metrics

    OpN --> Reports
    OpN --> Plots
```

## Component Interactions

### Operation Registry

```mermaid
graph TB
    subgraph "Registration"
        Op1[Operation Class 1]
        Op2[Operation Class 2]
        OpN[Operation Class N]
    end

    subgraph "Registry"
        Reg[OperationRegistry]
        Meta[Metadata Store]
        Dep[Dependency Graph]
    end

    subgraph "Discovery"
        API[API Request]
        Discovery[Discovery Service]
    end

    subgraph "Instantiation"
        Factory[Operation Factory]
        Instance[Operation Instance]
    end

    Op1 -->|@register_operation| Reg
    Op2 -->|@register_operation| Reg
    OpN -->|@register_operation| Reg

    Reg --> Meta
    Reg --> Dep

    API --> Discovery
    Discovery --> Reg
    Reg --> Factory
    Factory --> Instance
```

### Progress Tracking

```mermaid
graph TB
    subgraph "Hierarchy"
        Parent[Parent Progress]
        Child1[Child Progress 1]
        Child2[Child Progress 2]
        ChildN[Child Progress N]
    end

    subgraph "Tracking"
        Current[Current Progress]
        Total[Total Steps]
        Percent[Percentage]
    end

    subgraph "Reporting"
        Logger[Logger]
        Callback[Callback]
        UI[UI Update]
    end

    Parent --> Child1
    Parent --> Child2
    Parent --> ChildN

    Child1 --> Current
    Child2 --> Current
    ChildN --> Current

    Current --> Total
    Current --> Percent

    Current --> Logger
    Current --> Callback
    Current --> UI
```

### Caching Strategy

```mermaid
graph TB
    subgraph "Cache Key Generation"
        Input[Input Data Hash]
        Config[Config Hash]
        Key[Cache Key]
    end

    subgraph "Cache Storage"
        Memory[In-Memory Cache]
        Disk[Disk Cache]
    end

    subgraph "Cache Operations"
        Get[Get]
        Set[Set]
        Invalidate[Invalidate]
    end

    subgraph "Cache Policies"
        TTL[TTL Policy]
        LRU[LRU Policy]
        Size[Size Limit]
    end

    Input --> Key
    Config --> Key

    Key --> Get
    Get --> Memory
    Get --> Disk

    Set --> Memory
    Set --> Disk

    Invalidate --> Memory
    Invalidate --> Disk

    Memory --> TTL
    Disk --> LRU
    Disk --> Size
```

## Integration Patterns

### Dual Engine Support

```mermaid
graph LR
    subgraph "Data Input"
        Data[Input Data]
    end

    subgraph "Size Detection"
        Size[Size Analyzer]
    end

    subgraph "Engine Selection"
        Pandas[pandas Engine]
        Dask[Dask Engine]
    end

    subgraph "Operation"
        Op[Operation Execution]
    end

    Data --> Size
    Size -->|< 1M rows| Pandas
    Size -->|>= 1M rows| Dask

    Pandas --> Op
    Dask --> Op

    Op --> Result[Result]
```

### NLP Integration

```mermaid
graph TB
    subgraph "NLP Models"
        Spacy[spaCy Models]
        NLTK[NLTK Tokenizers]
        FastText[FastText Models]
    end

    subgraph "NLP Operations"
        Entity[Entity Recognition]
        Category[Category Matching]
        Lang[Language Detection]
    end

    subgraph "Caching"
        ModelCache[Model Cache]
        ResultCache[Result Cache]
    end

    Spacy --> Entity
    NLTK --> Entity
    FastText --> Category
    FastText --> Lang

    Entity --> ModelCache
    Category --> ResultCache
    Lang --> ResultCache
```

## References

- [system-architecture.md](./system-architecture.md) - Core system architecture
- [architecture-security.md](./architecture-security.md) - Security and performance architecture
- [code-standards.md](./code-standards.md) - Development guidelines
