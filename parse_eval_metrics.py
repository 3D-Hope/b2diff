"""Parse eval log files produced by the ThreedFront evaluation scripts.

Usage:
    python parse_eval_metrics.py <eval_log_file>

Prints a single comma-separated line:
    fid,kid,col_obj_pct,col_scene_pct,avg_num_obj,classifier_accuracy,classifier_std,kl_div
"""
import re
import sys


def parse_log(log_text: str) -> dict:
    metrics = {}

    # FID: 70.789...
    # m = re.search(r"FID:\s*([\d.]+)", log_text)
    # metrics["fid"] = m.group(1) if m else "N/A"

    # # KID: 0.00366...
    # m = re.search(r"KID:\s*([\d.eE+\-]+)", log_text)
    # metrics["kid"] = m.group(1) if m else "N/A"

    # Col_obj (percentage of objects that collide):
    #   Synthesized: 1290/3769 = 34.2266%
    m = re.search(
        r"Col_obj.*?Synthesized:\s*\d+/\d+\s*=\s*([\d.]+)%",
        log_text,
        re.DOTALL,
    )
    metrics["col_obj"] = m.group(1) if m else "N/A"

    # Col_scene (ratio of scenes with collisions):
    #   Synthesized: 529/1080 = 48.9815%
    m = re.search(
        r"Col_scene.*?Synthesized:\s*\d+/\d+\s*=\s*([\d.]+)%",
        log_text,
        re.DOTALL,
    )
    metrics["col_scene"] = m.group(1) if m else "N/A"

    # Obj (average number of objects per scene):
    #   Predicted layouts: 3.49
    m = re.search(r"Predicted layouts:\s*([\d.]+)", log_text)
    metrics["avg_num_obj"] = m.group(1) if m else "N/A"

    # "object category kl divergence: 0.08188316226005554"
    m = re.search(r"object category kl divergence:\s*([\d.eE+\-]+)", log_text, re.IGNORECASE)
    metrics["kl_div"] = m.group(1) if m else "N/A"

    # Mean tv_bed_reward: -1.7748
    # m = re.search(r"Mean tv_bed_reward:\s*([\d.eE+\-]+)", log_text)
    # metrics["mean_tv_bed_reward"] = m.group(1) if m else "N/A"

    # # Scenes with >1 tv_stand: 14.17 %
    # m = re.search(r"Scenes with >1 tv_stand:\s*([\d.]+)\s*%", log_text)
    # metrics["scenes_with_multiple_tv_stands"] = m.group(1) if m else "N/A"

    # # Scenes with >1 bed: 6.30 %
    # m = re.search(r"Scenes with >1 bed:\s*([\d.]+)\s*%", log_text)
    # metrics["scenes_with_multiple_beds"] = m.group(1) if m else "N/A"

    # out_of_bound_rate: 6.4349
    m = re.search(r"out_of_bound_rate:\s*([\d.eE+\-]+)", log_text)
    metrics["out_of_bound_rate"] = m.group(1) if m else "N/A"

    # walkable_average_rate, example line: Rwalkable: 0.8827717700101873
    m = re.search(r"Rwalkable:\s*([\d.eE+\-]+)", log_text)
    metrics["walkable_average_rate"] = m.group(1) if m else "N/A"

    # accessable_rate, example line: Rreach: 0.7765458991041984
    m = re.search(r"Rreach:\s*([\d.eE+\-]+)", log_text)
    metrics["accessable_rate"] = m.group(1) if m else "N/A"

    # box_wall_rate, example line: Rout: 0.4223255813953488
    m = re.search(r"Rout:\s*([\d.eE+\-]+)", log_text)
    metrics["box_wall_rate"] = m.group(1) if m else "N/A"

    # (1) Entropy of object category distribution (Synthesized only)
    m = re.search(
        r"Entropy of object category distribution:\s*\n\s*Synthesized:\s*([\d.eE+\-]+)",
        log_text,
    )
    metrics["object_category_entropy_synthesized"] = m.group(1) if m else "N/A"

    # (2) Average pairwise scene embedding distance (Synthesized only)
    m = re.search(
        r"Average pairwise scene embedding distance:\s*\n\s*Synthesized:\s*([\d.eE+\-]+)",
        log_text,
    )
    metrics["pairwise_scene_embedding_distance_synthesized"] = m.group(1) if m else "N/A"

    # (3) Furniture-weighted variance (Synthesized only)
    m = re.search(
        r"Furniture-weighted variance:\s*\n\s*Synthesized:\s*([\d.eE+\-]+)",
        log_text,
    )
    metrics["furniture_weighted_variance_synthesized"] = m.group(1) if m else "N/A"
    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_eval_metrics.py <eval_log_file>", file=sys.stderr)
        sys.exit(1)

    log_file = sys.argv[1]
    try:
        with open(log_file, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print("N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A")
        sys.exit(0)

    m = parse_log(content)
    print(
        ",".join(
            [
                m["col_obj"],
                m["col_scene"],
                m["avg_num_obj"],
                m["kl_div"],
                # m["mean_tv_bed_reward"],
                # m["scenes_with_multiple_tv_stands"],
                # m["scenes_with_multiple_beds"],
                m["out_of_bound_rate"],
                m["walkable_average_rate"],
                m["accessable_rate"],
                m["box_wall_rate"],
                m["object_category_entropy_synthesized"],
                m["pairwise_scene_embedding_distance_synthesized"],
                m["furniture_weighted_variance_synthesized"],
            ]
        )
    )


if __name__ == "__main__":
    main()
