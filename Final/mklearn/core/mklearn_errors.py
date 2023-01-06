class MKitLearnBasicError(Exception):
    """Base class for all MKitLearn errors."""
    pass


class MKitLearnNotImplementedError(MKitLearnBasicError, NotImplementedError):
    """Raised when a feature is not implemented yet."""
    pass


class ShapeError(MKitLearnBasicError, ValueError):
    """Raised when a shape is not valid."""
    pass
