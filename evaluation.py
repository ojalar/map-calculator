import os
import sys
import csv
import numpy as np
from itertools import chain, combinations

def loop_detection_files(path_detection, path_truth, iou, mode):
    # record of detections per class, either true positive 1 or false positive 0
    detections = dict()
    # record of false negatives per class
    false_negatives = dict()

    # loop through detection files, store results
    for r, d, f in os.walk(path_detection):
        for filename in f:
            if filename.endswith(".txt") or filename.endswith(".csv"):
                # acquire detection results
                new_detections, new_false_negatives =\
                    compare_files(path_detection + filename, path_truth + filename, iou, mode)

                # insert new detection results to the record
                for class_type, detection_result in new_detections.items():
                    # check if the class is in the previous detections, and add the results
                    if class_type in detections:
                        for result in detection_result:
                            detections.get(class_type).append(result)
                    else:
                        detections[class_type] = detection_result

                # insert new false negatives to the record
                for class_type, detection_result in new_false_negatives.items():
                    # check if the class is in the previous false_negatives, and add the results
                    if class_type in false_negatives:
                        false_negatives[class_type] = false_negatives.get(class_type) + detection_result
                    else:
                        false_negatives[class_type] = detection_result

    return detections, false_negatives

# compares the contents of a detection file and ground truth file
def compare_files(detection_file_path, truth_file_path, iou_threshold, mode):
    # initialize dictionaries for new detections
    new_detections = dict()
    new_false_negatives = dict()
    # read files
    detected_bbs, truth_bbs = read_files(detection_file_path, truth_file_path)

    #check if both ground truth and detection files were empty

    # create iou matrix for each class
    iou_matrices = generate_iou_matrices(detected_bbs, truth_bbs)

    # fill iou matrices
    iou_matrices, new_detections, new_false_negatives = \
        fill_iou_matrices(iou_matrices, new_detections, new_false_negatives,
        detected_bbs, truth_bbs)

    # map each detection to ground truth with maximum IOU above threshold
    new_detections, new_false_negatives = match_detections_and_ground_truths(iou_matrices,
        new_detections, new_false_negatives, iou_threshold, detected_bbs, truth_bbs, mode)

    return new_detections, new_false_negatives

# takes in an IOU matrix, matches detections with ground truths
def match_detections_and_ground_truths(iou_matrices, new_detections,
    new_false_negatives, iou_threshold, detected_bbs, truth_bbs, mode):
    # choose appropriate evaluation mode:
    # "normal" for normal AP evaluation
    # "modecla" for AP evaluation described in the paper, a single detection can
    # be responsible for (un-paired) ground truths if they all have IOU above threshold
    if mode == "normal":
        for classification in iou_matrices.keys():
            # loop through rows
            for i in range(iou_matrices.get(classification).shape[0]):
                # ground truth with maximum IOU linked, remove linked ground truths (columns)
                iou_candidates = np.argwhere(iou_matrices.get(classification)[i,:] >= iou_threshold)
                # check if IOUs over threshold, mark result
                if len(iou_candidates) == 0:
                    new_detections.get(classification).append((0, detected_bbs.get(classification)[i, 0]))
                else:
                    new_detections.get(classification).append((1, detected_bbs.get(classification)[i, 0]))
                    max_iou_ground_truth = np.argmax(iou_matrices.get(classification)[i,:])
                    # remove matched ground truth
                    iou_matrices[classification] = np.delete(iou_matrices.get(classification),
                        max_iou_ground_truth, 1)

            # check for remaining undetected ground truths
            if(iou_matrices.get(classification).shape[1] > 0):
                new_false_negatives[classification] += iou_matrices.get(classification).shape[1]

    # evaluation mode which links multiple ground truths to a single detection,
    # if they all individually surpass the IOU threshold
    elif mode == "multi":
        for classification in iou_matrices.keys():
            # loop through detections
            for i in range(iou_matrices.get(classification).shape[0]):
                # acquire indices of ground truths with IOU above threshold
                iou_candidates = np.argwhere(iou_matrices.get(classification)[i,:] >= iou_threshold)
                # check if IOUs over threshold, mark result
                # no IOU candidates - false positive
                if len(iou_candidates) == 0:
                    new_detections.get(classification).append((0, detected_bbs.get(classification)[i, 0]))
                # single IOU candidate, match detection and ground truth, remove ground truth
                elif len(iou_candidates) == 1:
                    new_detections.get(classification).append((1, detected_bbs.get(classification)[i, 0]))
                    max_iou_ground_truth = np.argmax(iou_matrices.get(classification)[i, :])
                    # remove matched ground truth
                    iou_matrices[classification] = np.delete(iou_matrices.get(classification),
                        max_iou_ground_truth, 1)
                # if multiple IOU candidates, procedure differs from normal.
                # maximum IOU candidate is first matched. if other candidates dont
                # have other detections to match them, they are matched to the
                # detection as well
                else:
                    sorted_iou_candidates = np.argsort(-iou_matrices.get(classification)[i,:])[:len(iou_candidates)]
                    # prepare list ground truths to be matched to the detection, match maximum IOU
                    matched = []
                    matched.append(sorted_iou_candidates[0])
                    new_detections.get(classification).append((1, detected_bbs.get(classification)[i, 0]))
                    # check other ground truths above IOU threshold, if they can be
                    # matched to other following detections
                    for index in sorted_iou_candidates[1:]:
                        matched_detections = np.argwhere(iou_matrices.get(classification)[:, index] >= iou_threshold)
                        following_detections = matched_detections[matched_detections > i]
                        # if they cannot be, match this ground truth to the original detection
                        if len(following_detections) == 0:
                            matched.append(index)
                            new_detections.get(classification).append((1, detected_bbs.get(classification)[i, 0]))
                        # else, loop and see if this ground truth is the highest IOU with a following detection.
                        # if it is not the highest IOU with any of the following detections, it is matched to the
                        # current detection
                        else:
                            highest_iou = False
                            for following_detection in following_detections:
                                if len(np.argwhere(iou_matrices.get(classification)[following_detection, :] > \
                                    iou_matrices.get(classification)[following_detection, index])) == 0:
                                    highest_iou = True
                                    break
                            if highest_iou == False:
                                matched.append(index)
                                new_detections.get(classification).append((1, detected_bbs.get(classification)[i, 0]))
                    # delete matched ground truths
                    iou_matrices[classification] = np.delete(iou_matrices.get(classification),
                        matched, 1)

            # check for remaining undetected ground truths
            if(iou_matrices.get(classification).shape[1] > 0):
                new_false_negatives[classification] += iou_matrices.get(classification).shape[1]

    # evaluation mode which takes highest IOU union of ground truths to fit to detection
    elif mode == "cluster":
        for classification in iou_matrices.keys():
            # loop through detections
            for i in range(iou_matrices.get(classification).shape[0]):
		# acquire indices of ground truths with IOU above 0
                iou_candidates = np.argwhere(iou_matrices.get(classification)[i,:] > 0)
                # if no such ground truths, false positive
                if len(iou_candidates) == 0:
                    new_detections.get(classification).append((0, detected_bbs.get(classification)[i, 0]))
                # otherwise ground truths with IOU above 0 found, remove ones that have the highest
                # IOU over threshold for another detection. highest IOU always included
                # for present detection
                else:
                    sorted_iou_candidates = np.argsort(-iou_matrices.get(classification)[i,:])[:len(iou_candidates)]
                    # prepare list ground truths to be matched to the detection, match maximum IOU
                    matched = []
                    matched.append(sorted_iou_candidates[0])
                    # check other ground truths with IOU above 0, if they can be
                    # matched to other following detections
                    for index in sorted_iou_candidates[1:]:
                        matched_detections = np.argwhere(iou_matrices.get(classification)[:, index] >= iou_threshold)
                        following_detections = matched_detections[matched_detections > i]
                        # if they cannot be, match this ground truth to the original detection
                        if len(following_detections) == 0:
                            matched.append(index)
                        # else, loop and see if this ground truth is the highest IOU with a following detection.
                        # if it is not the highest IOU with any of the following detections, it is matched to the
                        # current detection
                        else:
                            highest_iou = False
                            for following_detection in following_detections:
                                if len(np.argwhere(iou_matrices.get(classification)[following_detection, :] > \
                                    iou_matrices.get(classification)[following_detection, index])) == 0:
                                    highest_iou = True
                                    break
                            if highest_iou == False:
                                matched.append(index)

                    # find combination of matched ground truths with maximum iou
                    max_iou, max_combination = find_max_iou_combination(i, classification, detected_bbs, truth_bbs, matched)
                    if max_iou >= iou_threshold:
                        # mark a true positive for each ground truth
                        for _ in range(len(max_combination)):
                            new_detections.get(classification).append((1, detected_bbs.get(classification)[i, 0]))
                        # delete matched ground truths
                        iou_matrices[classification] = np.delete(iou_matrices.get(classification),
                            max_combination, 1)
                        # delete also from the original ground truth list, as it is utilized in IOU calculation
                        truth_bbs[classification] = np.delete(truth_bbs.get(classification),
                            max_combination, 0)
                    else:
                        new_detections.get(classification).append((0, detected_bbs.get(classification)[i, 0]))

            # check for remaining undetected ground truths
            if(iou_matrices.get(classification).shape[1] > 0):
                new_false_negatives[classification] += iou_matrices.get(classification).shape[1]

    return new_detections, new_false_negatives

# fills an IOU matrix with the IOUs of column index and row index
def fill_iou_matrices(iou_matrices, new_detections, new_false_negatives,
    detected_bbs, truth_bbs):
    # loop through ground truth bounding boxes per class, compare to the detections
    for classification in list(iou_matrices):
        # initialize dictionaries for new detections and new false negatives
        new_detections[classification] = []
        new_false_negatives[classification] = 0
        # if no detections for present class (false negatives), mark down number
        # delete iou matrix
        if iou_matrices.get(classification).shape[0] == 0:
            new_false_negatives[classification] += iou_matrices.get(classification).shape[1]
            del iou_matrices[classification]
        # if false positives are found for classes with no truth labels, append
        # 0s, remove iou matrix
        elif iou_matrices.get(classification).shape[1] == 0:
            for i in range(iou_matrices.get(classification).shape[0]):
                new_detections.get(classification).append((0, detected_bbs.get(classification)[i, 0]))
            del iou_matrices[classification]
        # calculate IOU for each pair of detections and ground truths
        else:
            for i in range(iou_matrices.get(classification).shape[0]):
                for j in range(iou_matrices.get(classification).shape[1]):
                    iou_matrices.get(classification)[i,j] = \
                        compare_bb(detected_bbs.get(classification)[i, 1:],
                        truth_bbs.get(classification)[j])

    return iou_matrices, new_detections, new_false_negatives

# read the detection and ground truth files of an image
def read_files(detection_file_path, truth_file_path):
    # insert detected bouding boxes into detected, true bounding boxes into truth
    with open(detection_file_path, newline = '') as f:
        reader = csv.reader(f, delimiter = " ", quoting=csv.QUOTE_NONNUMERIC)
        detected = list(reader)
    with open(truth_file_path, newline = '') as f:
        reader = csv.reader(f, delimiter = " ", quoting=csv.QUOTE_NONNUMERIC)
        truth = list(reader)

    detected = list(filter(len, detected))
    truth = list(filter(len, truth))
    detected = np.array([np.array(i) for i in detected])
    truth = np.array([np.array(i) for i in truth])
    # transform the lists into dictionaries, where the bounding boxes are
    # given for each class respectively. check for files
    if detected.shape[0] == 0:
        detected_bbs = dict()
    else:
        detected_bbs = {int(i): detected[detected[:, 0] == i, 1:] for i in np.unique(detected[:, 0])}
    if truth.shape[0] == 0:
        truth_bbs = dict()
    else:
        truth_bbs = {int(i): truth[truth[:, 0] == i, 1:] for i in np.unique(truth[:, 0])}

    return detected_bbs, truth_bbs

# generate iou matrices. each row is a detection, each column is a ground truth.
# elements are IOU
def generate_iou_matrices(detected_bbs, truth_bbs):
    iou_matrices = dict()
    for classification in truth_bbs.keys():
        # if no detections in the class, leave number of detections as zero
        if classification in detected_bbs.keys():
            iou_matrices[classification] = np.zeros((detected_bbs.get(classification).shape[0],
                truth_bbs.get(classification).shape[0]))
        else:
            iou_matrices[classification] = np.zeros((0,
                truth_bbs.get(classification).shape[0]))

    # check predictions for classes not present in the ground truths
    for classification in detected_bbs.keys():
        if classification in truth_bbs.keys():
            continue
        else:
            iou_matrices[classification] = np.zeros((detected_bbs.get(classification).shape[0], 0))

    return iou_matrices


# calculate IOU for two boxes
def compare_bb(box_1, box_2):
    #bb format: x1, y1, x2, y2
    x_a = max(box_1[0], box_2[0])
    y_a = max(box_1[1], box_2[1])
    x_b = min(box_1[2], box_2[2])
    y_b = min(box_1[3], box_2[3])

    # intersection area
    intersection_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # area of both boxes
    box_1_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box_2_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    # calculate iou
    iou = intersection_area / float(box_1_area + box_2_area - intersection_area)

    return iou

# find maximum iou combination of matched ground truths for a detection
def find_max_iou_combination(i, classification, detected_bbs, truth_bbs, matched):
    # initialize x and y boundaries
    min_x = detected_bbs.get(classification)[i, 1]
    min_y = detected_bbs.get(classification)[i, 2]
    max_x = detected_bbs.get(classification)[i, 3]
    max_y = detected_bbs.get(classification)[i, 4]

    # update boundaries according to matched ground truths
    for match in matched:
        if truth_bbs.get(classification)[match, 0] < min_x:
            min_x = truth_bbs.get(classification)[match, 0]
        if truth_bbs.get(classification)[match, 1] < min_y:
            min_y = truth_bbs.get(classification)[match, 1]
        if truth_bbs.get(classification)[match, 2] > max_x:
            max_x = truth_bbs.get(classification)[match, 2]
        if truth_bbs.get(classification)[match, 3] > max_y:
            max_y = truth_bbs.get(classification)[match, 3]

    # acquire all possible combinations of feasible ground truths for the current detection
    truth_combinations = chain.from_iterable(combinations(matched, r) for r in range(1, len(matched)+1))
    # find combination with maximum IOU
    max_iou = -1
    max_combination = -1
    for combination in truth_combinations:
        intersection = 0
        union = 0
        # loop through all pixels in defined x and y ranges
        for x in range(int(min_x), int(max_x) + 1):
            for y in range(int(min_y), int(max_y) + 1):
                inside_detection = 0
                inside_truth = 0
                # check whether the pixel is inside the detection
                if x >= detected_bbs.get(classification)[i, 1] and y >= detected_bbs.get(classification)[i, 2] \
                and x <= detected_bbs.get(classification)[i, 3] and y <= detected_bbs.get(classification)[i, 4]:
                    inside_detection = 1
                # check whether the pixel is inside a ground truth
                for index in combination:
                    if x >= truth_bbs.get(classification)[index, 0] and y >= truth_bbs.get(classification)[index, 1] \
                    and x <= truth_bbs.get(classification)[index, 2] and y <= truth_bbs.get(classification)[index, 3]:
                        inside_truth = 1
                        break

                # if inside both detection and ground truth: in intersection
                if inside_detection and inside_truth:
                    intersection += 1
                # if inside detection or ground truth: in union
                if inside_detection or inside_truth:
                    union += 1

        # update max IOU
        iou = intersection/union
        if iou > max_iou:
            max_iou = iou
            max_combination = combination

    return max_iou, max_combination
