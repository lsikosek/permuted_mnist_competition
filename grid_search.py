"""
Grid search for the Permuted MNIST agent.

Example:
python grid_search.py \
  --num-episodes 10 \
  --seed 42 \
  --lr-start 0.05 --lr-stop 0.20 --lr-step 0.05 \
  --epochs-start 10 --epochs-stop 30 --epochs-step 10 \
  --batch-start 64 --batch-stop 256 --batch-step 64 \
  --num-hidden-layers 3 \
  --hidden-start 128 128 64 \
  --hidden-stop  512 512 256 \
  --hidden-step   128 128 64 \
  --results-path grid_search_results.csv
"""

import argparse
import csv
import itertools
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from permuted_mnist.env.permuted_mnist import PermutedMNISTEnv
from agent import Agent


def float_range(start: float, stop: float, step: float) -> List[float]:
    """
    Inclusive float range.
    """
    if step <= 0:
        raise ValueError("step must be > 0")

    values = []
    current = start
    eps = step * 1e-9

    while current <= stop + eps:
        values.append(round(current, 10))
        current += step

    return values


def int_range(start: int, stop: int, step: int) -> List[int]:
    """
    Inclusive integer range.
    """
    if step <= 0:
        raise ValueError("step must be > 0")
    if stop < start:
        raise ValueError("stop must be >= start")

    return list(range(start, stop + 1, step))


def build_hidden_layer_grid(
    num_hidden_layers: int,
    hidden_start: Sequence[int],
    hidden_stop: Sequence[int],
    hidden_step: Sequence[int],
) -> List[Tuple[int, ...]]:
    """
    Builds all hidden-layer configurations.

    Example with num_hidden_layers=2:
      hidden_start=[128, 64]
      hidden_stop =[256,128]
      hidden_step =[128,64]

    Produces:
      (128, 64), (128, 128), (256, 64), (256, 128)
    """
    if not (len(hidden_start) == len(hidden_stop) == len(hidden_step) == num_hidden_layers):
        raise ValueError(
            "hidden-start, hidden-stop, and hidden-step must each have exactly "
            f"{num_hidden_layers} values"
        )

    per_layer_choices = []
    for i in range(num_hidden_layers):
        choices = int_range(hidden_start[i], hidden_stop[i], hidden_step[i])
        per_layer_choices.append(choices)

    return list(itertools.product(*per_layer_choices))


def evaluate_config(
    learning_rate: float,
    epochs: int,
    batch_size: int,
    hidden_layer_sizes: Sequence[int],
    num_episodes: int,
    seed: int,
    episode_timeout: float,
) -> dict:
    """
    Evaluate one hyperparameter configuration across all episodes.
    """
    env = PermutedMNISTEnv(number_episodes=num_episodes)
    env.set_seed(seed)

    accuracies = []
    episode_times = []
    total_time = 0.0

    for episode in range(num_episodes):
        task = env.get_next_task()
        if task is None:
            break

        agent = Agent(
            seed=seed,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            hidden_layer_sizes=hidden_layer_sizes,
        )

        start = time.time()
        agent.train(task["X_train"], task["y_train"])
        predictions = agent.predict(task["X_test"])
        elapsed = time.time() - start

        if elapsed > episode_timeout:
            raise TimeoutError(
                f"Episode {episode + 1} timed out: {elapsed:.2f}s > {episode_timeout:.2f}s"
            )

        accuracy = env.evaluate(predictions, task["y_test"])

        accuracies.append(float(accuracy))
        episode_times.append(float(elapsed))
        total_time += elapsed

    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_layer_sizes": tuple(hidden_layer_sizes),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_time": float(np.mean(episode_times)),
        "std_time": float(np.std(episode_times)),
        "total_time": float(total_time),
        "episodes_run": len(accuracies),
    }


def save_results_csv(results: List[dict], path: str) -> None:
    """
    Save all grid-search results to CSV.
    """
    if not results:
        return

    fieldnames = [
        "learning_rate",
        "epochs",
        "batch_size",
        "hidden_layer_sizes",
        "mean_accuracy",
        "std_accuracy",
        "mean_time",
        "std_time",
        "total_time",
        "episodes_run",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            row_to_write = dict(row)
            row_to_write["hidden_layer_sizes"] = ",".join(map(str, row["hidden_layer_sizes"]))
            writer.writerow(row_to_write)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for Permuted MNIST agent")

    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=60.0,
        help="Max allowed training+prediction time per episode in seconds",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="grid_search_results.csv",
        help="Where to save the CSV results",
    )

    # Learning rate range
    parser.add_argument("--lr-start", type=float, required=True)
    parser.add_argument("--lr-stop", type=float, required=True)
    parser.add_argument("--lr-step", type=float, required=True)

    # Epoch range
    parser.add_argument("--epochs-start", type=int, required=True)
    parser.add_argument("--epochs-stop", type=int, required=True)
    parser.add_argument("--epochs-step", type=int, required=True)

    # Batch size range
    parser.add_argument("--batch-start", type=int, required=True)
    parser.add_argument("--batch-stop", type=int, required=True)
    parser.add_argument("--batch-step", type=int, required=True)

    # Hidden layer search
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        required=True,
        help="Number of hidden layers to search over",
    )
    parser.add_argument(
        "--hidden-start",
        type=int,
        nargs="+",
        required=True,
        help="Start size for each hidden layer slot",
    )
    parser.add_argument(
        "--hidden-stop",
        type=int,
        nargs="+",
        required=True,
        help="Stop size for each hidden layer slot",
    )
    parser.add_argument(
        "--hidden-step",
        type=int,
        nargs="+",
        required=True,
        help="Step size for each hidden layer slot",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    learning_rates = float_range(args.lr_start, args.lr_stop, args.lr_step)
    epochs_list = int_range(args.epochs_start, args.epochs_stop, args.epochs_step)
    batch_sizes = int_range(args.batch_start, args.batch_stop, args.batch_step)

    hidden_layer_grid = build_hidden_layer_grid(
        num_hidden_layers=args.num_hidden_layers,
        hidden_start=args.hidden_start,
        hidden_stop=args.hidden_stop,
        hidden_step=args.hidden_step,
    )

    all_configs = list(
        itertools.product(
            hidden_layer_grid,
            learning_rates,
            epochs_list,
            batch_sizes,
        )
    )

    print(f"Total configurations to evaluate: {len(all_configs)}")
    print("-" * 100)

    results = []
    best_result = None

    try:
        for idx, (hidden_layer_sizes, learning_rate, epochs, batch_size) in enumerate(all_configs, start=1):
            print(
                f"[{idx}/{len(all_configs)}] "
                f"lr={learning_rate}, epochs={epochs}, batch={batch_size}, "
                f"hidden={hidden_layer_sizes}"
            )

            try:
                result = evaluate_config(
                    learning_rate=learning_rate,
                    epochs=epochs,
                    batch_size=batch_size,
                    hidden_layer_sizes=hidden_layer_sizes,
                    num_episodes=args.num_episodes,
                    seed=args.seed,
                    episode_timeout=args.episode_timeout,
                )
                results.append(result)

                results.sort(key=lambda x: x["mean_accuracy"], reverse=True)
                save_results_csv(results, args.results_path)

                print(
                    f"  -> mean_acc={result['mean_accuracy']:.4f}, "
                    f"std_acc={result['std_accuracy']:.4f}, "
                    f"mean_time={result['mean_time']:.2f}s"
                )

                if best_result is None or result["mean_accuracy"] > best_result["mean_accuracy"]:
                    best_result = result

            except TimeoutError as e:
                print(f"  -> skipped (timeout): {e}")
            except Exception as e:
                print(f"  -> skipped (error): {e}")

            print("-" * 100)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial results...")
        results.sort(key=lambda x: x["mean_accuracy"], reverse=True)
        save_results_csv(results, args.results_path)

    if not results:
        print("No successful runs.")
        return

    print("\nBest configuration so far:")
    best_result = max(results, key=lambda x: x["mean_accuracy"])
    print(
        f"  learning_rate     = {best_result['learning_rate']}\n"
        f"  epochs            = {best_result['epochs']}\n"
        f"  batch_size        = {best_result['batch_size']}\n"
        f"  hidden_layer_sizes= {best_result['hidden_layer_sizes']}\n"
        f"  mean_accuracy     = {best_result['mean_accuracy']:.4f}\n"
        f"  std_accuracy      = {best_result['std_accuracy']:.4f}\n"
        f"  mean_time         = {best_result['mean_time']:.2f}s\n"
        f"  results_csv       = {args.results_path}"
    )

if __name__ == "__main__":
    main()