# Implementation Plan for Categorical Support Utilities

## Phase 1: Foundation Utilities 

### 1.1 Create `text_processing_utils.py`
- Basic functions first: `normalize_text()`, `clean_category_name()`
- String similarity: `calculate_string_similarity()`, `find_closest_match()`
- Composite handling: `split_composite_value()`, `extract_tokens()`
- **Why first**: Core functionality needed by all other modules

### 1.2 Create `category_utils.py`
- Distribution analysis: `analyze_category_distribution()`, `calculate_category_entropy()`
- Rare category identification: `identify_rare_categories()`
- Basic grouping: `group_rare_categories()` (single_other strategy only)
- **Why second**: Needed for metrics and validation

## Phase 2: Core Infrastructure 

### 2.1 Create `hierarchy_dictionary.py`
- Basic class structure and initialization
- File loading via `pamola_core.utils.io`
- Simple lookup functionality
- Basic validation
- **Why third**: Depends on text_processing_utils

### 2.2 Update `validation_utils.py`
- Add `validate_categorical_field()`
- Add `validate_hierarchy_dictionary()`
- **Why fourth**: Needs category_utils for analysis

## Phase 3: Integration & Enhancement 

### 3.1 Update `metric_utils.py`
- Add `calculate_categorical_information_loss()`
- Add `calculate_generalization_height()`
- **Why fifth**: Requires all base utilities

### 3.2 Update `visualization_utils.py`
- Add `create_category_distribution_comparison()`
- Add `create_hierarchy_sunburst()` (if time permits)
- **Why sixth**: Lowest priority, nice-to-have

### 3.3 Enhance `hierarchy_dictionary.py`
- Add advanced features: fuzzy matching, batch operations
- Performance optimizations: caching, indexing
- **Why last**: Core functionality must work first


## Critical Path

```
text_processing_utils → category_utils → hierarchy_dictionary → validation_utils → metric_utils
                                    ↓
                              categorical_op.py
```

## Key Principles

1. **Start Simple**: Basic functionality first, optimize later
2. **Test Early**: Write tests alongside implementation
3. **Incremental Enhancement**: Get working version, then add features
4. **Dependencies First**: Build in dependency order
5. **Integration Last**: Ensure utilities work before integration

## Success Criteria per Phase

- **Phase 1**: Text normalization and category analysis working
- **Phase 2**: Dictionary loading and validation complete
- **Phase 3**: Metrics calculating, basic visualizations working
- **Phase 4**: Full operation functioning with all strategies