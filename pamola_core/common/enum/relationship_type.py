from enum import Enum

class RelationshipType(str, Enum):
    AUTO = "auto"
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"