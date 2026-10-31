"""ggufpeek — read a GGUF model file's header and metadata without loading it."""
from .gguf import GGUFError, GGUFFile, Tensor, read_gguf

__version__ = "0.1.0"
__all__ = ["read_gguf", "GGUFFile", "Tensor", "GGUFError", "__version__"]
