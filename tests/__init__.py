"""测试包。

运行方式（无需 pytest，标准库即可）::

    python -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import logging

# 测试会刻意触发各种告警路径（脏行、模板占位符错误等），属于预期行为，
# 不必打印到测试输出里。assertLogs 会临时接管 level，因此断言仍有效。
logging.getLogger("career_qa").setLevel(logging.ERROR)
