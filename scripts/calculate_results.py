import codecs
import os

import pandas as pd
from sklearn.metrics import precision_score, f1_score, recall_score, classification_report, roc_curve, auc, \
    precision_recall_curve
import matplotlib.pyplot as plt
METRICS = {"Precision": precision_score, "Recall": recall_score,
           "F-score": f1_score, }


def main():
    true_labels_path = r"corpora/corpus_ruen_normalized/test.tsv"
    predicted_labels_path = r"test_labels/bilingual_pretrain_ruen/labels_pretrain_ruen_5.tsv"
    output_dir = r"test_labels/results/bilingual_pretrain_ruen/"
    output_fname = r"results.tsv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_fname)

    true_labels_and_tweets_df = pd.read_csv(true_labels_path, sep="\t",quoting=3,quotechar=None, encoding="utf-8")
    print(true_labels_and_tweets_df)
    true_labels_df = true_labels_and_tweets_df["class"]
    print(true_labels_df)

    predicted_labels_df = pd.read_csv(predicted_labels_path, sep="\t", encoding="utf-8", header=None)
    print(predicted_labels_df)
    predicted_positive_probs_df = predicted_labels_df.iloc[:, [1]]
    print(predicted_positive_probs_df)
    predicted_labels_df = predicted_labels_df.idxmax(axis=1)
    # fpr, tpr, thresholds = roc_curve(true_labels_df, predicted_positive_probs_df, )
    #
    # import matplotlib.pyplot as plt
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()

    prec, rec, thresholds = precision_recall_curve(true_labels_df, predicted_positive_probs_df, )
    f_scores = []
    print(len(prec))
    print(len(rec))
    print(len(thresholds))
    for i in range(len(prec) - 1):
        precision = prec[i]
        recall = rec[i]
        f_score = 2 * precision * recall / (precision + recall + 1e-10)
        f_scores.append(f_score)
    plt.plot(thresholds, f_scores, color='darkorange', )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('F-score')
    plt.ylabel('Threshold')
    plt.title('F-measure')
    plt.legend(loc="lower right")
    plt.show()


    # lw = 2
    # plt.plot(rec, prec, color='darkorange',
    #          lw=lw, label='PR curve (area = %0.2f)' % auc(rec, prec))
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend(loc="lower right")
    # plt.show()

    results = {}

    with codecs.open(output_path, "a+", encoding="utf-8") as output_file:
        print(classification_report(true_labels_df, predicted_labels_df))
        for metric_name, metric in METRICS.items():
            results[metric_name] = metric(true_labels_df, predicted_labels_df)
            print(f"{metric_name}", metric(true_labels_df, predicted_labels_df))
        #output_file.write(",".join(results.keys()))
        #output_file.write("\n")
        output_file.write(",".join([str(x) for x in results.values()]))
        output_file.write('\n')

    mismatch_df = true_labels_and_tweets_df
    mismatch_df["pred_y"] = predicted_labels_df
    mismatch_df = mismatch_df[mismatch_df["pred_y"] != mismatch_df['class']]
    print(mismatch_df)
    mismatch_df.to_csv("MISMATCH_DF.tsv", sep='\t')

if __name__ == '__main__':
    main()
