import numpy as np

import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=plt.cm.Greys):
    # Colormaps: jet, Greys
    cm_normalized = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)

    # Show confidences
    for i, cas in enumerate(cm): 
        for j, c in enumerate(cas): 
            if c > 0: 
                plt.text(j-0.1, i+0.2, c, fontsize=16, fontweight='bold', color='#b70000')

    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show(block=True)

def plot_roc_curve(test_target, pred_score): 
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr['micro'], tpr['micro'], _ = metrics.roc_curve(test_target, np.max(pred_score, axis=1))
    roc_auc['micro'] = metrics.auc(fpr['micro'], tpr['micro'])

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'], label='micro-average ROC curve (area = {0:0.2f})'
             ''.format(metrics.roc_auc['micro']))

    # plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show(block=True)

    # for i in range(n_classes):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])

# python metrics.py --classifier ~/data/results/plot/classifier.h5

if __name__ == "__main__": 

    import argparse
    import os.path
    from bot_utils.db_utils import AttrDict

    # Parse directory
    parser = argparse.ArgumentParser(
        description='Run UW RGB-D training')
    parser.add_argument(
        '-c', '--classifier', type=str, required=True, 
        help="Classifier path")
    args = parser.parse_args()

    clf_path = os.path.expanduser(args.classifier)
    # try: 

    cinfo = AttrDict.load(clf_path)

    cm = metrics.confusion_matrix(cinfo.test_target, cinfo.pred_target)
    # print ' Confusion matrix (Test): %s' % (cm)
    print ' Accuracy score (Test): %4.3f' % (metrics.accuracy_score(cinfo.test_target, cinfo.pred_target))
    print ' Report (Test):\n %s' % (metrics.classification_report(cinfo.test_target, cinfo.pred_target, 
                                                                  target_names=cinfo.target_names))

    # plot_roc_curve(cinfo.test_target, cinfo.pred_score)

    # except: 
    #     raise RuntimeError('Error loading classfier %s' % clf_path )




    # rtarget_names = cinfo.target_names
    # for idx, t in enumerate(rtarget_names): 
    #     if t == 'MICRO': rtarget_names[idx] = 'MICROWAVE'
    #     if t == 'PROJECTOR': rtarget_names[idx] = 'SCREEN'

    # # Show confusion matrix
    # plot_confusion_matrix(cm, rtarget_names)

    # print 'Training/Testing for %i classes' % (len(cinfo.target_names))        
    



