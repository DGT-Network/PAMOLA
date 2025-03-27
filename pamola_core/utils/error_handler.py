import wrapt
import sys


class PamolaError(Exception):
    """
    A base error class for this application.
    """
    pass


class ArgumentValidationError(PamolaError):
    def __init__(self, validation_messages):
        self.validation_messages = validation_messages


class ValidationError(PamolaError):
    def __init__(self, message, field):
        self.message = message
        self.field = field
        super().__init__("Validation Error: {}, {}".format(field, message))


class AnonymizationError(PamolaError):
    def __init__(self, message):
        self.message = message
        super().__init__("Anonymization Error: {}".format(message))


@wrapt.decorator
def intercept_exception(wrapped, instance, args, kwargs):
    """Handle Exceptions."""
    try:
        return wrapped(*args, **kwargs)
    except AssertionError as ae:
        print(ae, file=sys.stderr)
        sys.exit(1)
    except Exception as ge:
        print(
            "Unknown error. To debug run with env var LOG_LEVEL=DEBUG",
        )
        raise ge
