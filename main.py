import os
import sys
from pathlib import Path

if __name__ == '__main__':
    config = Path(sys.argv[1])
    assert config.exists()
    if config.is_file():
        print(f'python run.py {str(config)}')
        os.system(f'python run.py {str(config)}')
        sys.exit(0)  # 0: successful termination.
    os.system(f'ls {str(config)}')
    for cfg in config.glob('*.json'):
        print(f'python run.py {str(cfg)}')
        os.system(f'python run.py {str(cfg)}')
