"""支持 ``python -m career_qa`` 方式启动。"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
