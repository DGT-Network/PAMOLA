from enum import Enum

class RelationshipType(str, Enum):
    AUTO = "auto"
    ONE_TO_ONE = "one-to-one"
    ONE_TO_MANY = "one-to-many"