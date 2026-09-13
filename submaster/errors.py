class SubmasterError(Exception):
    """Base exception raised for user-facing CLI failures."""


class ModelResponseError(SubmasterError):
    """A model response is unusable and may be retried with a smaller input."""
