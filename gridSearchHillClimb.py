import copy
import itertools
import math
import time
import numpy as np

from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
from agent2 import Agent


# ----------------------------
# Helpers
# ----------------------------

def canonical_config(cfg: dict) -> tuple:
    """Turn config dict into a hashable key."""
    items = []
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, list):
            v = tuple(v)
        items.append((k, v))
    return tuple(items)


def pretty_config(cfg: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def make_env(num_episodes: int, seed: int) -> PermutedMNISTEnv:
    env = PermutedMNISTEnv(number_episodes=num_episodes)
    env.set_seed(seed)
    return env


def evaluate_config(
    cfg: dict,
    num_episodes: int = 3,
    env_seed: int = 42,
    hard_time_limit: float = 60.0,
    verbose: bool = False,
) -> dict:
    """
    Evaluate one Agent config over a fixed number of episodes.
    Uses the same interaction pattern as eval.py:
      agent = Agent(...)
      agent.train(X_train, y_train)
      predictions = agent.predict(X_test)
    """
    env = make_env(num_episodes=num_episodes, seed=env_seed)

    accuracies = []
    times = []

    for episode in range(num_episodes):
        task = env.get_next_task()
        if task is None:
            break

        agent = Agent(**cfg)

        start = time.time()
        agent.train(task["X_train"], task["y_train"])
        predictions = agent.predict(task["X_test"])
        elapsed = time.time() - start

        if elapsed > hard_time_limit:
            return {
                "config": copy.deepcopy(cfg),
                "mean_accuracy": -1.0,
                "std_accuracy": 0.0,
                "mean_time": elapsed,
                "timed_out": True,
                "episodes": episode + 1,
            }

        accuracy = env.evaluate(predictions, task["y_test"])
        accuracies.append(float(accuracy))
        times.append(float(elapsed))

        if verbose:
            print(
                f"    episode {episode + 1:2d}: "
                f"acc={accuracy:.4f}, time={elapsed:.2f}s"
            )

    return {
        "config": copy.deepcopy(cfg),
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else -1.0,
        "std_accuracy": float(np.std(accuracies)) if accuracies else 0.0,
        "mean_time": float(np.mean(times)) if times else float("inf"),
        "timed_out": False,
        "episodes": len(accuracies),
    }


def generate_base_grid() -> list[dict]:
    """
    Focused grid over the most plausible MLP settings.
    These are biased toward 'strong but still trainable in < 60s on CPU'.
    """
    hidden_layer_options = [
        [768, 768],
        [1024, 1024],
        [1536, 1536],
        [1024, 1024, 1024],
        [1536, 1536, 1536],
        [2048, 2048],
    ]

    learning_rates = [0.03, 0.06, 0.10, 0.15]
    batch_sizes = [512, 1024, 2048]
    weight_decays = [0.0, 1e-5, 1e-4]
    epochs_list = [80]

    grid = []
    for hidden_layer_sizes, learning_rate, batch_size, weight_decay, epochs in itertools.product(
        hidden_layer_options,
        learning_rates,
        batch_sizes,
        weight_decays,
        epochs_list,
    ):
        cfg = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "time_limit_seconds": 54.0,
            "num_threads": 2,
            "seed": 123,
        }
        grid.append(cfg)

    return grid


def neighbor_values_numeric(value, multipliers, min_value=None, max_value=None, as_int=False):
    out = set()
    for m in multipliers:
        v = value * m
        if as_int:
            v = int(round(v))
        if min_value is not None:
            v = max(min_value, v)
        if max_value is not None:
            v = min(max_value, v)
        out.add(v)
    return sorted(out)


def generate_neighbors(cfg: dict) -> list[dict]:
    """
    Local hill-climbing neighborhood around a promising config.
    """
    neighbors = []

    base_widths = list(cfg["hidden_layer_sizes"])
    depth = len(base_widths)
    mean_width = int(round(sum(base_widths) / len(base_widths)))

    width_candidates = neighbor_values_numeric(
        mean_width,
        multipliers=[0.75, 0.9, 1.0, 1.1, 1.25],
        min_value=512,
        max_value=4096,
        as_int=True,
    )

    depth_candidates = sorted(set([max(2, depth - 1), depth, min(4, depth + 1)]))

    lr_candidates = neighbor_values_numeric(
        cfg["learning_rate"],
        multipliers=[0.5, 0.75, 1.0, 1.25, 1.5],
        min_value=0.01,
        max_value=0.3,
        as_int=False,
    )

    batch_candidates = neighbor_values_numeric(
        cfg["batch_size"],
        multipliers=[0.5, 1.0, 2.0],
        min_value=256,
        max_value=4096,
        as_int=True,
    )

    wd_candidates = sorted(set([
        0.0,
        cfg["weight_decay"],
        max(0.0, cfg["weight_decay"] * 0.1),
        cfg["weight_decay"] * 10.0 if cfg["weight_decay"] > 0 else 1e-5,
    ]))

    epoch_candidates = neighbor_values_numeric(
        cfg["epochs"],
        multipliers=[0.67, 1.0, 1.5],
        min_value=40,
        max_value=300,
        as_int=True,
    )

    for width in width_candidates:
        for d in depth_candidates:
            for lr in lr_candidates:
                for bs in batch_candidates:
                    for wd in wd_candidates:
                        for ep in epoch_candidates:
                            neighbor = copy.deepcopy(cfg)
                            neighbor["hidden_layer_sizes"] = [width] * d
                            neighbor["learning_rate"] = float(lr)
                            neighbor["batch_size"] = int(bs)
                            neighbor["weight_decay"] = float(wd)
                            neighbor["epochs"] = int(ep)
                            neighbors.append(neighbor)

    # Deduplicate
    dedup = {}
    for n in neighbors:
        dedup[canonical_config(n)] = n
    return list(dedup.values())


def hill_climb(
    start_cfg: dict,
    start_score: float,
    cache: dict,
    probe_episodes: int = 3,
    env_seed: int = 42,
    max_rounds: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Greedy hill climbing:
    - evaluate neighborhood
    - move to best strictly better config
    - repeat a few rounds
    """
    best_result = cache[canonical_config(start_cfg)]
    current_cfg = copy.deepcopy(start_cfg)
    current_score = start_score

    for round_idx in range(max_rounds):
        if verbose:
            print(f"\n  Hill climb round {round_idx + 1} from score {current_score:.4f}")

        neighbors = generate_neighbors(current_cfg)
        improving_candidates = []

        for neighbor in neighbors:
            key = canonical_config(neighbor)
            if key not in cache:
                cache[key] = evaluate_config(
                    neighbor,
                    num_episodes=probe_episodes,
                    env_seed=env_seed,
                    verbose=False,
                )

            result = cache[key]
            if result["timed_out"]:
                continue

            score = result["mean_accuracy"]
            if score > current_score:
                improving_candidates.append(result)

        if not improving_candidates:
            break

        improving_candidates.sort(
            key=lambda r: (r["mean_accuracy"], -r["mean_time"]),
            reverse=True,
        )
        best_neighbor = improving_candidates[0]

        current_cfg = copy.deepcopy(best_neighbor["config"])
        current_score = best_neighbor["mean_accuracy"]

        if current_score > best_result["mean_accuracy"]:
            best_result = best_neighbor

        if verbose:
            print(
                f"    moved to: acc={best_neighbor['mean_accuracy']:.4f}, "
                f"time={best_neighbor['mean_time']:.2f}s | "
                f"{pretty_config(best_neighbor['config'])}"
            )

    return best_result


# ----------------------------
# Main search
# ----------------------------

def main():
    # Search controls
    probe_episodes = 3
    full_eval_episodes = 10
    env_seed = 42

    # Thresholds
    promising_threshold = 0.98
    target_threshold = 0.99

    # Cache every config we evaluate
    cache: dict[tuple, dict] = {}

    # Base grid
    base_grid = generate_base_grid()

    # Optional: sort configs so likely winners are checked first
    def heuristic_priority(cfg: dict):
        width = sum(cfg["hidden_layer_sizes"]) / len(cfg["hidden_layer_sizes"])
        depth = len(cfg["hidden_layer_sizes"])
        # prefer medium-large models, not the craziest ones
        width_penalty = abs(width - 1536)
        depth_penalty = abs(depth - 2)
        lr_penalty = abs(cfg["learning_rate"] - 0.06)
        bs_penalty = abs(cfg["batch_size"] - 1024) / 1024
        return (width_penalty + 300 * depth_penalty + 1000 * lr_penalty + 500 * bs_penalty)

    base_grid.sort(key=heuristic_priority)

    print(f"Base grid size: {len(base_grid)}")

    all_results = []
    promising_results = []

    for idx, cfg in enumerate(base_grid, start=1):
        key = canonical_config(cfg)
        if key not in cache:
            result = evaluate_config(
                cfg,
                num_episodes=probe_episodes,
                env_seed=env_seed,
                verbose=False,
            )
            cache[key] = result
        else:
            result = cache[key]

        all_results.append(result)

        print(
            f"[{idx:3d}/{len(base_grid)}] "
            f"acc={result['mean_accuracy']:.4f}, "
            f"time={result['mean_time']:.2f}s, "
            f"timeout={result['timed_out']} | "
            f"{pretty_config(cfg)}"
        )

        if not result["timed_out"] and result["mean_accuracy"] >= promising_threshold:
            print("  -> promising, starting hill climb")
            climbed = hill_climb(
                start_cfg=cfg,
                start_score=result["mean_accuracy"],
                cache=cache,
                probe_episodes=probe_episodes,
                env_seed=env_seed,
                max_rounds=3,
                verbose=True,
            )
            promising_results.append(climbed)

    # Combine all screened results and hill-climbed results
    candidate_pool = all_results + promising_results
    candidate_pool = [r for r in candidate_pool if not r["timed_out"]]

    # Deduplicate by config and keep best recorded score for each config
    best_by_cfg = {}
    for r in candidate_pool:
        key = canonical_config(r["config"])
        if key not in best_by_cfg or r["mean_accuracy"] > best_by_cfg[key]["mean_accuracy"]:
            best_by_cfg[key] = r

    ranked = sorted(
        best_by_cfg.values(),
        key=lambda r: (r["mean_accuracy"], -r["mean_time"]),
        reverse=True,
    )

    print("\nTop screened candidates:")
    for r in ranked[:10]:
        print(
            f"  acc={r['mean_accuracy']:.4f}, time={r['mean_time']:.2f}s | "
            f"{pretty_config(r['config'])}"
        )

    # Full evaluation on the best few
    finalists = ranked[:5]
    print(f"\nRunning full evaluation on top {len(finalists)} candidates...")

    final_results = []
    for r in finalists:
        cfg = r["config"]
        full_result = evaluate_config(
            cfg,
            num_episodes=full_eval_episodes,
            env_seed=env_seed,
            verbose=False,
        )
        final_results.append(full_result)

        print(
            f"FULL acc={full_result['mean_accuracy']:.4f} +/- {full_result['std_accuracy']:.4f}, "
            f"time={full_result['mean_time']:.2f}s | "
            f"{pretty_config(cfg)}"
        )

        if full_result["mean_accuracy"] >= target_threshold:
            print("\nTarget exceeded.")
            break

    final_results.sort(
        key=lambda r: (r["mean_accuracy"], -r["mean_time"]),
        reverse=True,
    )

    if final_results:
        best = final_results[0]
        print("\nBest final config:")
        print(f"  mean acc : {best['mean_accuracy']:.4f}")
        print(f"  std acc  : {best['std_accuracy']:.4f}")
        print(f"  mean time: {best['mean_time']:.2f}s")
        print(f"  config   : {pretty_config(best['config'])}")
    else:
        print("\nNo valid final candidates found.")


if __name__ == "__main__":
    main()