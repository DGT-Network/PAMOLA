from pydantic import *
from email_validator import EmailNotValidError
from typing import TypeVar
V = TypeVar("V")


def validate_type(value, t: V):
    return type(value) is t


def email_is_valid(value):
    try:
        validate_email(value)
        return True
    except EmailNotValidError:
        return False

def validate_range(value, min_val, max_val):
    return min_val <= value <= max_val