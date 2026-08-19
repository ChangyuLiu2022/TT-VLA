from pathlib import Path
import yaml
from datetime import datetime

def main():
    stats = {}

    # old stats
    sold_path = Path(__file__).parent.parent / "scripts" / "stats"
    solds = sold_path.glob("stats-*.yaml")
    solds = sorted(list(solds), key=lambda x: x.name)
    for s in solds:
        print(f"{s.name}")
    for sold in solds:
        cfg = yaml.safe_load(sold.read_text())
        for load_path, envs in cfg.items():
            if load_path not in stats:
                stats[load_path] = {}
            for env_name, seeds in envs.items():
                if env_name not in stats[load_path]:
                    stats[load_path][env_name] = {}
                for seed, stat in seeds.items():
                    if seed not in stats[load_path][env_name]:
                        stats[load_path][env_name][seed] = {}
                    stats[load_path][env_name][seed].update(stat)

    # wandb
    wandb_path = Path(__file__).parent.parent / "wandb"
    runs = wandb_path.glob("offline-run-*")

    for run in runs:
        cfg = yaml.safe_load((run / "glob" / "config.yaml").read_text())

        load_path = "/".join(cfg["vla_load_path"].split("/")[-3:])
        env_name = cfg["env_id"]
        seed = cfg["seed"]

        if load_path not in stats:
            stats[load_path] = {}
        if env_name not in stats[load_path]:
            stats[load_path][env_name] = {}

        train_vis_dir = run / "glob" / "vis_0_train" / "stats.yaml"
        if train_vis_dir.exists():
            train_stats = yaml.safe_load(train_vis_dir.read_text())
            if "stats" in train_stats:
                if "train" not in stats[load_path][env_name]:
                    stats[load_path][env_name]["train"] = {}
                stats[load_path][env_name]["train"][seed] = train_stats["stats"]
                stats[load_path][env_name]["train"][seed]["path"] = str(run)

        test_vis_dir = run / "glob" / "vis_0_test" / "stats.yaml"
        if test_vis_dir.exists():
            test_stats = yaml.safe_load(test_vis_dir.read_text())
            if "stats" in test_stats:
                if "test" not in stats[load_path][env_name]:
                    stats[load_path][env_name]["test"] = {}
                stats[load_path][env_name]["test"][seed] = test_stats["stats"]
                stats[load_path][env_name]["test"][seed]["path"] = str(run)

    # save stats
    tt = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(__file__).parent.parent / "scripts" / "stats" / f"stats-{tt}.yaml"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)

if __name__ == "__main__":
    main()
