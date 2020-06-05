import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import running_precision_and_recall

# show the precision-recall plot for a class
def visualise_results(classification, class_detections, class_false_negatives,
    smoothed_precision, recall_samples):
    mpl.style.use("seaborn")
    #plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('font', size=18)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    plt.rc('legend', fontsize=14)    # legend fontsize
    plt.rc('figure', titlesize=18)  # fontsize of the figure title

    running_precision, running_recall = running_precision_and_recall(class_detections,
        class_false_negatives)

    #initialize figure
    plt.figure(figsize=(8, 6))

    # plot the normal precision - recall curve
    plt.plot(running_recall, running_precision, label = "Precision-recall")
    if len(smoothed_precision) > 1:
        plt.plot(np.concatenate((np.array([recall_samples[0]]), np.repeat(recall_samples[1:-1], 2),
            np.array([recall_samples[-1]]))), np.repeat(smoothed_precision,2), label = "Monotonic precision-recall", linestyle = "-.")

    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.title("Precision-recall for class: " + str(classification))
    plt.legend()
    plt.tight_layout()
    plt.xlim(0,1.1)
    plt.ylim(0,1.1)
    plt.show()
