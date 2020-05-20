import json
import os
import shutil
import sys
from pathlib import Path

from src.slack import slack_message

if __name__ == "__main__":
    config = Path(sys.argv[1])
    assert config.exists()
    # Grid search
    if config.name.endswith(".grid.json"):
        with open(config, "r") as f:
            grid = json.load(f)
        grid_search_target = []
        for key, val in grid.items():
            if type(val) is list:
                grid_search_target.append(key)
        assert len(grid_search_target) <= 2, "Grid search only supports <=2 so far."
        if len(grid_search_target) == 1:
            target_name = grid_search_target[0]
            grid_combinations = [{target_name: v} for v in grid[target_name]]
        elif len(grid_search_target) == 2:
            grid_combinations = []
            target_name1 = grid_search_target[0]
            target_name2 = grid_search_target[1]
            for v1 in grid[target_name1]:
                for v2 in grid[target_name2]:
                    grid_combinations.append(
                        {target_name1: v1, target_name2: v2,}
                    )
        config_name_path = config.parent / config.name[:-10]
        if config_name_path.exists():
            # Danger: remove directory with files in it.
            shutil.rmtree(config_name_path)
        config_name_path.mkdir(exist_ok=False)
        for grid_options in grid_combinations:
            new_grid = grid.copy()
            name_suffix = ""
            for k, v in grid_options.items():
                new_grid[k] = v
                name_suffix += k + str(v)
            name = name_suffix + ".auto.json"
            with open(config_name_path / name, "w") as f:
                json.dump(new_grid, f, indent=2)
        config = config_name_path
    # Single config
    if config.is_file():
        slack_message("Start {}".format(config))
        print(f"python run.py {str(config)}")
        os.system(f"python run.py {str(config)}")
        sys.exit(0)  # 0: successful termination.
    # Manual comparison
    os.system(f"ls {str(config)}")
    slack_message(
        "Start grid\n{}".format("\n".join([str(c) for c in config.iterdir()]))
    )
    for cfg in config.glob("*.json"):
        print(f"python run.py {str(cfg)}")
        os.system(f"python run.py {str(cfg)}")
