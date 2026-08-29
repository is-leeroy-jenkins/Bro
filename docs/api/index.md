# API Reference

Bro is primarily a Streamlit application, so its API documentation distinguishes between
**import-safe modules** and `app.py`.

- [App](app.md) documents the major functional surfaces without importing the Streamlit entry point.
- [Configuration](config.md) uses mkdocstrings for `config` when the module is import-safe.
- [Logging](boogr.md) uses mkdocstrings for `boogr` when available.

This arrangement reduces documentation-build failures caused by importing a top-level Streamlit UI
module during `mkdocs build`.
