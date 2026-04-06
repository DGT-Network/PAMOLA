# DateTime Generalization Enumerations

**Module:** `pamola_core.common.enum.datetime_generalization`
**Version:** 1.0
**Status:** Stable
**Last Updated:** 2026-03-23

## Overview

This module provides two related enumerations for protecting privacy in temporal data:
- **DatetimeMethod**: Strategies for generalizing datetime values
- **DatePeriod**: Granularity levels for datetime decomposition

Together they enable flexible datetime anonymization with privacy-utility tradeoff control.

## DatetimeMethod Enumeration

Defines the techniques available for generalizing datetime values while preserving temporal patterns and utility.

### Members

| Member | Value | Description |
|--------|-------|-------------|
| `RANGE` | `"range"` | Group dates into ranges or time intervals |
| `PERIOD` | `"period"` | Extract period (year, month, etc.) from date |
| `FORMAT` | `"format"` | Change date format or representation |

### Usage

```python
from pamola_core.common.enum.datetime_generalization import DatetimeMethod

method = DatetimeMethod.PERIOD
print(method.value)  # Output: "period"
```

### Member Descriptions

#### RANGE
**Value:** `"range"`

Groups dates into broader time intervals, suppressing exact dates. Dates within same range become indistinguishable.

**Privacy Mechanism:**
- Generalizes precise dates to intervals
- Increases indistinguishability within range
- Range width controls privacy level

**Example:**
```
Original: 2024-01-03, 2024-01-15, 2024-01-28
Generalized (Q1 2024): All map to "Q1 2024"
```

**Use cases:**
- Event timestamps (transaction date to quarter)
- Medical records (appointment date to month)
- Activity logs (precise time to day)

**Privacy Impact:** High (dates become indistinguishable)

**Utility Impact:** Moderate (temporal ordering may be lost)

#### PERIOD
**Value:** `"period"`

Extracts a specific component (year, month, day of week, hour) from the date. Loses finer-grained temporal information.

**Privacy Mechanism:**
- Removes specific date information
- Retains only selected temporal component
- Component selection controls detail level

**Example:**
```
Original: 2024-01-15 14:30:45
Period=MONTH: 01
Period=YEAR: 2024
Period=WEEKDAY: Monday
```

**Use cases:**
- Birth month for age analysis (not exact birthdate)
- Day of week for pattern analysis (not specific date)
- Hour of day for temporal patterns (not exact time)

**Privacy Impact:** High (depends on period granularity)

**Utility Impact:** Moderate (patterns preserved)

#### FORMAT
**Value:** `"format"`

Changes the representation or format of dates without losing the underlying temporal information. Useful for obfuscation or standardization.

**Privacy Mechanism:**
- Reformats date without changing data
- May obscure original representation
- Works with date formats enum

**Example:**
```
Original: 01/15/2024
Reformatted: 2024-01-15
Reformatted: January 15, 2024
```

**Use cases:**
- Standardizing disparate date formats
- Obscuring original representation
- Regional format normalization

**Privacy Impact:** Low (information preserved)

**Utility Impact:** High (no information loss)

## DatePeriod Enumeration

Defines the granularity levels for extracting temporal components from datetime values. Used with PERIOD method.

### Members

| Member | Value | Description | Example |
|--------|-------|-------------|---------|
| `YEAR` | `"year"` | Calendar year | 2024 |
| `QUARTER` | `"quarter"` | Quarter (Q1-Q4) | Q1, Q2, Q3, Q4 |
| `MONTH` | `"month"` | Month of year | 01, 02, ..., 12 |
| `WEEKDAY` | `"weekday"` | Day of week | Monday, Tuesday, ... |
| `HOUR` | `"hour"` | Hour of day | 0-23 |

### Usage

```python
from pamola_core.common.enum.datetime_generalization import DatePeriod

period = DatePeriod.MONTH
print(period.value)  # Output: "month"

# Extract month from date
dates = pd.Series(pd.date_range('2024-01-01', periods=3))
months = dates.dt.month  # [1, 1, 1]
```

### Member Descriptions

#### YEAR
**Value:** `"year"`

Extracts only the year from a datetime. Hides month, day, time information.

**Privacy:** Very high
**Utility:** Low (loses seasonal patterns)

**Example:**
```
2024-01-15 10:30 → 2024
2024-12-31 23:59 → 2024
```

**Use cases:**
- Decade-level analysis
- Historical trend only
- Maximum privacy requirement

#### QUARTER
**Value:** `"quarter"`

Extracts quarter (Q1-Q4) of year. Preserves seasonal patterns while hiding months.

**Privacy:** High
**Utility:** Moderate (seasonal patterns visible)

**Example:**
```
2024-01-15 → Q1
2024-06-30 → Q2
2024-12-31 → Q4
```

**Use cases:**
- Business quarterly analysis
- Seasonal trends without monthly detail
- Financial reporting aggregation

#### MONTH
**Value:** `"month"`

Extracts the month (1-12) from date. Hides day and time information.

**Privacy:** Moderate-High
**Utility:** Moderate-High (daily patterns hidden, monthly visible)

**Example:**
```
2024-01-03 10:30 → 01
2024-01-15 14:45 → 01
2024-02-28 22:15 → 02
```

**Use cases:**
- Monthly aggregation
- Seasonal analysis
- Hide specific dates

#### WEEKDAY
**Value:** `"weekday"`

Extracts day of week (Monday-Sunday or 0-6). Useful for weekly pattern analysis.

**Privacy:** Moderate (loses date info, shows patterns)
**Utility:** Moderate (weekly patterns visible)

**Example:**
```
2024-01-01 (Monday) → Monday
2024-01-15 (Monday) → Monday
2024-01-16 (Tuesday) → Tuesday
```

**Use cases:**
- Work schedule patterns
- Weekly behavior analysis
- Day-of-week effects

#### HOUR
**Value:** `"hour"`

Extracts hour (0-23) from time. Hides minutes, seconds, and date information.

**Privacy:** Moderate (shows temporal pattern)
**Utility:** Moderate-High (time-of-day patterns visible)

**Example:**
```
2024-01-15 10:30:45 → 10
2024-01-15 10:45:00 → 10
2024-01-15 11:00:00 → 11
```

**Use cases:**
- Peak hour analysis
- Hourly traffic/usage patterns
- Time-of-day effects

## Selection Guide

### By Privacy Requirement

**Maximum Privacy**
- Use `RANGE` with large intervals, or
- Use `PERIOD` with `YEAR` or `QUARTER`

**Strong Privacy**
- Use `PERIOD` with `MONTH` or `WEEKDAY`
- Use `RANGE` with weekly/monthly intervals

**Moderate Privacy**
- Use `PERIOD` with `HOUR` or `WEEKDAY`
- Use `FORMAT` (minimal privacy)

### By Utility Requirement

**Preserve Temporal Patterns**
- `PERIOD` with `HOUR`, `WEEKDAY`, or `MONTH`
- Maintains daily/weekly/monthly trends

**Preserve Yearly Trends**
- `PERIOD` with `QUARTER` or `YEAR`
- `RANGE` with yearly intervals

**Preserve Exact Timestamps**
- `FORMAT` only (information preserved)

### By Data Type

**Transaction Dates**
- Use `RANGE` with monthly boundaries
- Or `PERIOD` with `MONTH`

**Appointment Times**
- Use `PERIOD` with `HOUR` and `MONTH`
- Or `RANGE` with daily intervals

**Event Dates**
- Use `PERIOD` with `YEAR` or `QUARTER`
- Or `RANGE` with quarterly intervals

## Common Patterns

### Multi-Level Generalization

```python
from pamola_core.common.enum.datetime_generalization import DatetimeMethod, DatePeriod
import pandas as pd

def apply_datetime_generalization(dates: pd.Series,
                                 method: DatetimeMethod,
                                 period: DatePeriod = None) -> pd.Series:
    """Apply datetime generalization."""
    if method == DatetimeMethod.RANGE:
        # Group into ranges
        return pd.cut(dates, bins='M')  # Monthly ranges

    elif method == DatetimeMethod.PERIOD:
        if period == DatePeriod.YEAR:
            return dates.dt.year
        elif period == DatePeriod.MONTH:
            return dates.dt.month
        elif period == DatePeriod.WEEKDAY:
            return dates.dt.day_name()
        elif period == DatePeriod.HOUR:
            return dates.dt.hour

    elif method == DatetimeMethod.FORMAT:
        return dates.dt.strftime('%Y-%m-%d')  # ISO format

dates = pd.date_range('2024-01-01', periods=5, freq='D')
generalized = apply_datetime_generalization(dates, DatetimeMethod.PERIOD, DatePeriod.MONTH)
```

### Privacy-Aware Selection

```python
from pamola_core.common.enum.datetime_generalization import DatetimeMethod, DatePeriod

def select_generalization(privacy_level: str):
    """Choose generalization based on privacy requirement."""
    if privacy_level == "high":
        return DatetimeMethod.PERIOD, DatePeriod.YEAR
    elif privacy_level == "moderate":
        return DatetimeMethod.PERIOD, DatePeriod.MONTH
    else:
        return DatetimeMethod.FORMAT, None
```

## Best Practices

1. **Use Enums for Type Safety**
   ```python
   # Good
   method = DatetimeMethod.PERIOD
   period = DatePeriod.MONTH

   # Avoid
   method = "period"  # String is error-prone
   ```

2. **Document Generalization Choice**
   ```python
   # Good - explains reasoning
   # Hide exact dates but preserve monthly trends
   method = DatetimeMethod.PERIOD
   period = DatePeriod.MONTH
   ```

3. **Validate Before Processing**
   ```python
   if not pd.api.types.is_datetime64_any_dtype(dates):
       dates = pd.to_datetime(dates)
   ```

4. **Handle Timezone Considerations**
   ```python
   # Preserve or standardize timezone
   if dates.dt.tz is not None:
       dates = dates.dt.tz_convert('UTC')
   ```

## Related Documentation

- [Common Module Overview](../common_overview.md)
- [Enumerations Quick Reference](./enums_reference.md)
- [Numeric Generalization](./numeric_generalization.md)

## Changelog

**v1.0 (2026-03-23)**
- Initial documentation
- DatetimeMethod: RANGE, PERIOD, FORMAT
- DatePeriod: YEAR, QUARTER, MONTH, WEEKDAY, HOUR
- Privacy-utility tradeoff analysis
