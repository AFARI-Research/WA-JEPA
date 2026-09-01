"""Test package marker.

Without it ``tests`` is only a namespace package, and any regular ``tests`` package
installed in site-packages shadows it regardless of ``sys.path`` order, so the commands
in the README fail with ``ModuleNotFoundError: No module named 'tests.test_...'``.
"""
