"""python -m tts_trainer 入口，转交 CLI 主函数。 / Entry for python -m tts_trainer, delegating to the CLI."""
from .cli import main


raise SystemExit(main())
