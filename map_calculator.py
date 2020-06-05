import sys
import os
import argparse
import numpy as np

from evaluation import loop_detection_files
from visualisation import visualise_results
from utils import precision
from utils import recall
from utils import AP

#-------------IMPORTANT-------------
# Assumption is made that the detections are ordered highest confidence first.
#-----------------------------------

# Detections should be listed as text or csv file per image in a single folder.
# Ground truths should be listed in a single folder with identical names.

# detection format: <class> <confidence> <x1> <y1> <x2> <y2>
# ground truth format: <class> <x1> <y1> <x2> <y2>

def main(path_detection, path_truth, mode, iou):
    print("Path to detections:", path_detection)
    print("Path to ground truth:", path_truth)
    print("Chosen mode:", mode)
    print("IoU-threshold:", iou)

    # acquire detections and false_negatives for each class
    # detections: {class: [(1,0.95), (0,0.87), (0,0.6), (1,0.67), ...., (0,0.9), (1,0.7)]}
    # false_negatives: int
    detections, false_negatives = loop_detection_files(path_detection, path_truth, iou, mode)
    # store APs for calculating mAP
    average_precisions = []
    # analyse results of each class
    for classification in detections.keys():
        classification_detections = [d[0] for d in sorted(detections.get(classification), key=lambda tup: tup[1], reverse=True)]
        # print precision, recall, AP and show precision-recall curve
        print("-------------")
        print("CLASS", classification)
        print("Precision:", precision(classification_detections))
        print("Recall:", recall(classification_detections,
            false_negatives.get(classification)))
        average_precision, smoothed_precision, recall_samples = AP(classification, classification_detections,
            false_negatives.get(classification))
        average_precisions.append(average_precision)
        print("AP:", average_precision)
        visualise_results(classification, classification_detections,
            false_negatives.get(classification), smoothed_precision, recall_samples)
        print("-------------")
    print("********")
    print("mAP:", np.mean(average_precisions))
    print("********")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detection",
	   help="path to the folder containing the detections")
    ap.add_argument("-t", "--truth",
	   help="path to the folder containing the ground truth")
    ap.add_argument("-m", "--mode",
	   help="which evaluation mode is used: 'normal', 'multi' or 'cluster'")
    ap.add_argument("-i", "--iou",
	   help="what is the IOU threshold: 0-1")
    args = vars(ap.parse_args())

    if not args.get("detection", False):
        print("No path provided to the detection folder")
        sys.exit()
    if not args.get("truth", False):
        print("No path provided to the ground truth folder")
        sys.exit()

    if not args.get("mode", False):
        mode = "normal"
    else:
        if args.get("mode") == "normal":
            mode = "normal"
        elif args.get("mode") == "multi":
            mode = "multi"
        elif args.get("mode") == "cluster":
            mode = "cluster"
        else:
            print("Invalid mode selected")
            sys.exit()

    if not args.get("iou", False):
        iou = 0.5
    else:
        iou = float(args.get("iou"))

    main(args.get("detection"), args.get("truth"), mode, iou)
