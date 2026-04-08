# Root conftest.py — D:\FUCK!!\Grad\Code\api\conftest.py
# This runs before pytest touches any test file or subfolder conftest.
import sys
import pathlib

_root = pathlib.Path(__file__).resolve().parent  # = D:\FUCK!!\Grad\Code\api
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
