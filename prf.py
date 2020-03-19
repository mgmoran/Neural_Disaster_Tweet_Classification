from collections import defaultdict
import mxnet as mx
import numpy as np

"""A class that computes precision, recall and F1 values given a model's output. """

class PRF():
    def __init__(self,model, dataloader, do_curve):
        self.model = model
        self.dataloader = dataloader
        self.do_curve = do_curve
    def evaluate(self):
        counts = defaultdict(lambda: defaultdict(int))
        prf = defaultdict(lambda: defaultdict(float))
        total = 0
        all_labels = [] #all
        labels = []  # store the ground truth labels
        preds = []  # store model predictions
        scores = [] # store raw model scores
        for i, (data, label) in enumerate(self.dataloader):
            labels.append(label)
            out = self.model(data)
            pred = mx.nd.softmax(out)
            preds.append(pred)
            for j in range(out.shape[0]):  ## out.shape[0] refers to the batch size
                total += 1
                lab = int(labels[i][j].asscalar())
                all_labels.append(lab)
                pred = np.argmax(preds[i][j]).asscalar()
                pos_prob = preds[i][j][1].asscalar()
                scores.append(pos_prob)
                if lab == pred:
                    counts[lab]['truepos'] += 1
                else:
                    counts[lab]['falseneg'] +=1
                    counts[pred]['falsepos'] += 1
        print(total)
        acc =  sum([counts[lab]['truepos'] for lab in counts])/ total
        for lab in counts:
            truepos = counts[lab]['truepos']
            falseneg = counts[lab]['falseneg']
            falsepos = counts[lab]['falsepos']
            try:
                prf[lab]['prec'] = truepos / (truepos + falsepos)
            except ZeroDivisionError:
                prf[lab]['prec'] = 0
            try:
                prf[lab]['rec'] = truepos / (truepos + falseneg)
            except ZeroDivisionError:
                prf[lab]['rec'] = 0
            try:
                prf[lab]['f1'] = (2 * (prf[lab]['prec'] *  prf[lab]['rec'])) / (prf[lab]['prec'] + prf[lab]['rec'])
            except ZeroDivisionError:
                prf[lab]['f1'] = 0
        if self.do_curve:
            return acc, prf, scores, all_labels
        else:
            return acc, prf


