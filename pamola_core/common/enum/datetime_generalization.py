from enum import Enum

class DatetimeMethod(Enum):
    RANGE = "range"
    PERIOD = "period"
    FORMAT = "format"

class DatePeriod(Enum):
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    WEEKDAY = "weekday"
    HOUR = "hour"