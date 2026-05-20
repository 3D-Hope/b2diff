"""Script to count the number of out of boundary objects and pairwise bounding 
boxes IoU in predicted layouts.
"""
import argparse
import numpy as np
import pickle

from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from threed_front.evaluation.utils import count_out_of_boundary, compute_bbox_iou


def _get_raw_dataset_from_results(threed_front_results):
    """Resolve dataset for scene_idx lookup while preserving old behavior.

    Preferred source is the dataset embedded in the results object. This keeps
    scene indices consistent for filtered GT exports. If unavailable/incompatible,
    fall back to rebuilding from config as before.
    """
    embedded_dataset = getattr(threed_front_results, "test_dataset", None)
    if embedded_dataset is not None and hasattr(embedded_dataset, "get_room_params"):
        print("Using embedded dataset from results object for scene_idx lookup.")
        return embedded_dataset

    config = threed_front_results.config
    return get_raw_dataset(
        config["data"],
        split=config["validation"].get("splits", ["test"]),
        include_room_mask=True,
    )


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    parser.add_argument(
        "result_file",
        help="Path to a pickled result file (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--erosion",
        default=0.1,
        type=float,
        help="Amount of erosion in meters from predicted sizes (default: 0.1)"
    )
    parser.add_argument(
        "--area_tol",
        default=1e-5,
        type=float,
        help="Maximum out of boundary bbox area to be considered within floor bound (default: 1e-5)"
    )
    args = parser.parse_args(argv)
    assert args.erosion >= 0
    assert args.area_tol >= 0

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    assert threed_front_results.floor_condition

    # Load dataset (embedded dataset if available, fallback to old config path)
    raw_dataset = _get_raw_dataset_from_results(threed_front_results)
    
    # Count number of out-of-boundary objects
    oob_objects_total, num_objects_total = 0, 0             # synthesized
    oob_objects_ref_total, num_objects_ref_total = 0, 0     # real

    # Compute pairwise bounding boxes IoU
    bbox_iou_total = []             # synthesized
    inter_pairs_total = []          # synthesized, number of intersected (i.e. positive iou)
    bbox_iou_ref_total = []         # real
    inter_pairs_ref_total = []      # real, number of intersected (i.e. positive iou)
    
    # Collision metrics for paper
    colliding_objects_total = 0     # synthesized, total count of objects involved in collisions
    scenes_with_collision_total = 0 # synthesized, count of scenes with at least one collision
    colliding_objects_ref_total = 0 # real, total count of objects involved in collisions
    scenes_with_collision_ref_total = 0 # real, count of scenes with at least one collision

    for scene_idx, scene_layout in threed_front_results:
        gt_scene_layout = raw_dataset.get_room_params(scene_idx)
        
        # Out-of-boundary objects
        # synthesized layout
        oob_objects, oob_mask = count_out_of_boundary(
            gt_scene_layout["fpbpn"][:, :2],
            scene_layout, 
            erosion=args.erosion, 
            area_tol=args.area_tol
        )
        oob_objects_total += oob_objects
        num_objects_total += len(oob_mask)
        # check ground-truth layout
        oob_objects, oob_mask = count_out_of_boundary(
            gt_scene_layout["fpbpn"][:, :2],
            gt_scene_layout, 
            erosion=args.erosion, 
            area_tol=args.area_tol
        )
        oob_objects_ref_total += oob_objects
        num_objects_ref_total += len(oob_mask)

        # Bounding boxes IoU
        # synthesized layout
        bbox_iou = np.array(compute_bbox_iou(scene_layout))
        if len(bbox_iou) == 0:
            bbox_iou_total.append(0)
            inter_pairs_total.append(0)
        else:   
            bbox_iou_total.append(bbox_iou.mean()) 
            inter_pairs_total.append((bbox_iou>0).sum())
            
            # Count objects involved in collisions (for Col_obj metric)
            if (bbox_iou > 0).any():
                num_objects = len(scene_layout["class_labels"])
                colliding_objects_set = set()
                
                # Map 1D IoU array back to (i, j) pairs
                pair_idx = 0
                for i in range(num_objects):
                    for j in range(i+1, num_objects):
                        if bbox_iou[pair_idx] > 0:
                            colliding_objects_set.add(i)
                            colliding_objects_set.add(j)
                        pair_idx += 1
                
                colliding_objects_total += len(colliding_objects_set)
                scenes_with_collision_total += 1
        
        # ground-truth layout
        bbox_iou = np.array(compute_bbox_iou(gt_scene_layout))
        bbox_iou_ref_total.append((bbox_iou).mean()) 
        inter_pairs_ref_total.append((bbox_iou>0).sum())
        
        # Count objects involved in collisions for ground truth (for Col_obj metric)
        if len(bbox_iou) > 0 and (bbox_iou > 0).any():
            num_objects_ref = len(gt_scene_layout["class_labels"])
            colliding_objects_ref_set = set()
            
            # Map 1D IoU array back to (i, j) pairs
            pair_idx = 0
            for i in range(num_objects_ref):
                for j in range(i+1, num_objects_ref):
                    if bbox_iou[pair_idx] > 0:
                        colliding_objects_ref_set.add(i)
                        colliding_objects_ref_set.add(j)
                    pair_idx += 1
            
            colliding_objects_ref_total += len(colliding_objects_ref_set)
            scenes_with_collision_ref_total += 1
    print(f"out_of_bound_rate: {oob_objects_total/num_objects_total * 100:.4f}")

    print("(1) Found {} out-of-boundary objects from {} total ({:.4f} %) in {} synthesized scenes."\
          .format(
              oob_objects_total, num_objects_total, 
              oob_objects_total/num_objects_total * 100, len(threed_front_results)
            ))
    print("    For reference, there are {} out-of-boundary objects from {} total ({:.4f} %) in "
          "corresponding ground-turth scenes."\
          .format(
              oob_objects_ref_total, num_objects_ref_total,
              oob_objects_ref_total/num_objects_ref_total * 100
            ))
    print("(2) Average number of intersected bbox paris is {}, average IoU is {:.4f} % over {} synthesized scenes."\
          .format(
              np.mean(inter_pairs_total), np.mean(bbox_iou_total) * 100, 
              len(threed_front_results)
            ))
    print("    For reference, these are {} and {:.4f} % in ground-truth scenes."\
            .format(
                np.mean(inter_pairs_ref_total), np.mean(bbox_iou_ref_total) * 100
            ))
    
    # Calculate and print collision metrics as defined in the paper
    col_obj_synth = (colliding_objects_total / num_objects_total * 100) if num_objects_total > 0 else 0
    col_scene_synth = (scenes_with_collision_total / len(threed_front_results) * 100) if len(threed_front_results) > 0 else 0
    col_obj_ref = (colliding_objects_ref_total / num_objects_ref_total * 100) if num_objects_ref_total > 0 else 0
    col_scene_ref = (scenes_with_collision_ref_total / len(threed_front_results) * 100) if len(threed_front_results) > 0 else 0
    
    print("\n" + "="*80)
    print("COLLISION METRICS (as per paper definition)")
    print("="*80)
    print("(3) Col_obj (percentage of objects that collide with other objects):")
    print("    Synthesized: {}/{} = {:.4f}%".format(
        colliding_objects_total, num_objects_total, col_obj_synth))
    print("    Ground-truth: {}/{} = {:.4f}%".format(
        colliding_objects_ref_total, num_objects_ref_total, col_obj_ref))
    
    print("(4) Col_scene (ratio of scenes that possess object collisions):")
    print("    Synthesized: {}/{} = {:.4f}%".format(
        scenes_with_collision_total, len(threed_front_results), col_scene_synth))
    print("    Ground-truth: {}/{} = {:.4f}%".format(
        scenes_with_collision_ref_total, len(threed_front_results), col_scene_ref))
    print("="*80)


if __name__ == "__main__":
    main(None)
# Col_scene (20.99% vs 72.22%): How many scenes have ANY collision
# Col_obj (10.67% vs 42.52%): How many objects are involved in collisions
# Avg intersected pairs (0.28 vs 1.46): How many pairs are colliding per scene on average
# Avg IoU (0.10% vs 0.26%): How much overlap there is on average