import numpy as np

def precision(class_detections):
    return np.sum(class_detections)/len(class_detections)

def recall(class_detections, class_false_negatives):
    return np.sum(class_detections)/(np.sum(class_detections) + class_false_negatives)

# AUC (Area Under Curve) version of AP
def AP(classification, class_detections, class_false_negatives):
    # acquire running precision and running recall
    running_precision, running_recall = running_precision_and_recall(class_detections,
        class_false_negatives)

    # smooth the zigzag pattern, track drops and acquire corresponding recall values
    smoothed_precision = []
    recall_samples = []
    
    # start recall samples from zero, smoothed precision from the maximum
    recall_samples.append(0)
    smoothed_precision.append(np.max(running_precision))
    previous_max_precision_right = np.max(running_precision)
    for i in range(len(running_precision)):
        max_precision_right = np.max(running_precision[i:])
        if max_precision_right != previous_max_precision_right:
            smoothed_precision.append(max_precision_right)
            recall_samples.append(running_recall[i])
        previous_max_precision_right = max_precision_right
    recall_samples.append(running_recall[-1])

    # use smoothed precision and recall samples to calculate AUC
    average_precision = np.sum(smoothed_precision * np.diff(recall_samples))

    return average_precision, smoothed_precision, recall_samples

# compute running precision and recall for precision-recall curve and AP
def running_precision_and_recall(class_detections, class_false_negatives):
    running_precision = []
    running_recall = []
    recall_denominator = np.sum(class_detections) + class_false_negatives
    # loop through detections, append detections one by one to running_precision
    # and running_recall
    for i in range(1, len(class_detections) + 1):
        running_precision.append(np.sum(class_detections[:i])/i)
        running_recall.append(np.sum(class_detections[:i])/recall_denominator)

    return running_precision, running_recall
