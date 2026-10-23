
"""DataMind Agents Package.

Agent modules are imported on demand to reduce application startup overhead.
"""

__all__ = [
    "run_detective",
    "run_analyst",
    "run_ml_engineer",
    "run_reporter",
]


def __getattr__(name):
    if name == "run_detective":
        from .detective import run_detective
        return run_detective

    if name == "run_analyst":
        from .analyst import run_analyst
        return run_analyst

    if name == "run_ml_engineer":
        from .ml_engineer import run_ml_engineer
        return run_ml_engineer

    if name == "run_reporter":
        from .reporter import run_reporter
        return run_reporter

    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
