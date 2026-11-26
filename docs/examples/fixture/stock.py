"""Small inventory source fixture for static analysis."""


def item_count(items: list[str]) -> int:
    """Return the number of items in a batch."""
    return len(items)


class Inventory:
    """Hold the size of an inventory batch."""

    def is_empty(self, count: int) -> bool:
        """Report whether the batch count is zero."""
        return count == 0
