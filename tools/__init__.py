"""
tools — Pure Python tool modules. No AG2 imports allowed here.

Architecture Reference: architecture.md § 12.1 (Hard Boundary Rule)

The AG2-facing public API per module is always a flat function.
OOP implementation (Strategy, Template Method, etc.) lives beneath it.
AG2 never sees classes — only callables.
"""
