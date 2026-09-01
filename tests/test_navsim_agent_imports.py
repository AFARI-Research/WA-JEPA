import sys
import types
import unittest
from pathlib import Path


def _load_import_helper():
    source = Path("eval/navsim_agent.py").read_text(encoding="utf-8")
    start = source.index("_REPO_ROOT =")
    end = source.index("\n_prepare_worldmodel_imports()")
    namespace = {"Path": Path, "sys": sys, "__file__": str(Path("eval/navsim_agent.py"))}
    exec(source[start:end], namespace)
    return namespace["_prepare_worldmodel_imports"], namespace["_REPO_ROOT"]


class NavsimAgentImportsTest(unittest.TestCase):
    def test_prepare_worldmodel_imports_removes_foreign_utils_module(self):
        prepare, repo_root = _load_import_helper()
        fake_utils = types.ModuleType("utils")
        old_path = list(sys.path)
        old_utils = sys.modules.get("utils")
        try:
            sys.path[:] = ["/third_party"]
            sys.modules["utils"] = fake_utils

            prepare()

            self.assertEqual(sys.path[0], repo_root)
            self.assertNotIn("utils", sys.modules)
        finally:
            sys.path[:] = old_path
            if old_utils is None:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = old_utils


if __name__ == "__main__":
    unittest.main()
