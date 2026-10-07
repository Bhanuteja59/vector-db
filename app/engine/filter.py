from typing import Any, Dict

def evaluate_condition(field_value: Any, condition: Any) -> bool:
    """Evaluate a single field against a condition (scalar or operator dict)."""
    if not isinstance(condition, dict):
        return field_value == condition

    for op, target in condition.items():
        if op == "$eq" and field_value != target:
            return False
        elif op == "$ne" and field_value == target:
            return False
        elif op == "$gt":
            if field_value is None or field_value <= target:
                return False
        elif op == "$gte":
            if field_value is None or field_value < target:
                return False
        elif op == "$lt":
            if field_value is None or field_value >= target:
                return False
        elif op == "$lte":
            if field_value is None or field_value > target:
                return False
        elif op == "$in":
            if not isinstance(target, (list, tuple, set)) or field_value not in target:
                return False
        elif op == "$nin":
            if isinstance(target, (list, tuple, set)) and field_value in target:
                return False
    return True

def matches_filter(metadata: Dict[str, Any], query_filter: Dict[str, Any] | None) -> bool:
    """
    Evaluate if a metadata dictionary matches a query filter expression.
    Supports:
      - Field matching: {"category": "books"}
      - Operators: {"rating": {"$gte": 4.5}}
      - Logical: {"$and": [...]}, {"$or": [...]}
    """
    if not query_filter:
        return True

    for key, val in query_filter.items():
        if key == "$and":
            if not isinstance(val, list):
                return False
            if not all(matches_filter(metadata, sub_cond) for sub_cond in val):
                return False
        elif key == "$or":
            if not isinstance(val, list):
                return False
            if not any(matches_filter(metadata, sub_cond) for sub_cond in val):
                return False
        elif key == "$not":
            if matches_filter(metadata, val):
                return False
        else:
            field_value = metadata.get(key)
            if not evaluate_condition(field_value, val):
                return False

    return True
