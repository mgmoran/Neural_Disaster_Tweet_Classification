# coding: utf-8

from argparse import ArgumentParser
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
from load_data import load_dataset
from models import CNNTextClassifier, DANTextClassifier
from prf import PRF
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve, average_precision_score

parser = ArgumentParser()
parser.add_argument('--train_file', type=str, default= 'new_train.tsv', help='File containing file representing the input TRAINING data')
parser.add_argument('--val_file', type=str, default = 'new_val.tsv', help='File containing file representing the input VALIDATION data')
parser.add_argument('--test_file', type=str, default='new_test.tsv', help='File containing file representing the input TEST data')
parser.add_argument('--exp_prefix',type=str,default=None)
parser.add_argument('--epochs', type=int, default=None, help='Upper epoch limit')
parser.add_argument('--optimizer',type=str, help='Optimizer (adam, sgd, etc.)', default='adam')
parser.add_argument('--lr',type=float, help='Learning rate', default=0.001)
parser.add_argument('--batch_size',type=int, help='Training batch size', default=16)
parser.add_argument('--dropout', type=float, help='Dropout ratio', default=0.5)
parser.add_argument('--embedding_source', type=str, default='glove.twitter.27B.100d', help='Pre-trained embedding source name')
parser.add_argument('--fix_embedding', action='store_true', help='Fix embedding vectors instead of fine-tuning them')
parser.add_argument('--lstm', action='store_true', help='Whether or not the model uses lstm layers')
parser.add_argument('--model_type',type=str, help='CNN or DAN model',default='CNN')

args = parser.parse_args()
for arg in vars(args):
    print(arg, getattr(args, arg))

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

def train_classifier(vocabulary, data_train, data_val, data_test, ctx=mx.cpu()):

    ## set up the data loaders for each data source
    train_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dataloader   = mx.gluon.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=True)
    test_dataloader  = mx.gluon.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=True)


    emb_input_dim, emb_output_dim = vocabulary.embedding.idx_to_vec.shape

    if args.model_type=='CNN':
        model = CNNTextClassifier(emb_input_dim, emb_output_dim, dropout=args.dropout,
                                  lstm=args.lstm)
    else:
        model = DANTextClassifier(emb_input_dim, emb_output_dim, dropout=args.dropout)

    model.initialize(ctx=ctx)  ## initialize model parameters on the context ctx
    model.embedding.weight.set_data(vocab.embedding.idx_to_vec) ## set the embedding layer parameters to the pre-trained embedding in the vocabulary

    if args.fix_embedding:
        model.embedding.collect_params().setattr('grad_req', 'null')

    model.hybridize() ## OPTIONAL for efficiency - perhaps easier to comment this out during debugging
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': args.lr})
    epoch = 0

    # keeps track of validation accuracy at each epoch, for early stopping
    val_acc_increase = True
    val_acc = 0
    while val_acc_increase:
        print(f"epoch = {epoch}")
        epoch_cum_loss = 0
        for i, (data, label) in enumerate(train_dataloader):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with torch.no_grad():
                output = model(data)
                l = loss_fn(output, label).mean()  # get the average loss over the batch
            l.backward()
            trainer.step(label.shape[0]) ## update weights
            epoch_cum_loss += l.asscalar()  ## needed to convert mx.nd.array value back to Python float
        evaluate(model, val_dataloader)
        train_acc, prf_train = evaluate(model, train_dataloader)
        last_val_acc = val_acc
        val_acc, prf_val = evaluate(model, val_dataloader)
        if last_val_acc > val_acc:
            val_acc_increase = False
        print(f"epoch_cum_loss={epoch_cum_loss:.2f}")
        print(f"train accuracy={train_acc:.2f}")
        print(f"val accuracy={val_acc:.2f}")
        epoch += 1

    test_acc, prf_test, scores, labels  = evaluate(model, test_dataloader, do_curve=True)
    average_precision = average_precision_score(labels, scores)
    prec, rec, thresholds = precision_recall_curve(labels, scores, pos_label=None, sample_weight=None)

    ### prints out readable stats of the model and results on test set if a file prefix
    ### is specified.
    if args.exp_prefix:
        pyplot.plot(rec, prec, marker='.', label=args.exp_prefix)
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.legend()
        pyplot.savefig(args.exp_prefix + ".png")
        with open(args.exp_prefix + ".txt",'w+') as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")
            f.write("\n\n")
            f.write(f"test accuracy={test_acc:.4f}\n")
            f.write(f"average precision = {average_precision}\n")
            f.write(f"test metrics =\n")
            f.write("\tprec\trec\t\tf1\n")
            for lab in prf_test:
                f.write(f"{lab}\t{prf_test[lab]['prec']:.2f}\t{prf_test[lab]['rec']:.2f}\t{prf_test[lab]['f1']:.2f}\n")

def evaluate(model, dataloader, do_curve=False, ctx=mx.cpu()):
    """
    Get predictions on the dataloader items from model
    Creates an instance of the PRF class and calls its evaluate method.
    If do_curve is True, produces a Precision-Recall curve.
    """
    metrics = PRF(model, dataloader, do_curve)
    return metrics.evaluate()

def main():
    vocab, train_dataset, val_dataset, test_dataset = load_dataset(args.train_file, args.val_file, args.test_file)
    glove_twitter = nlp.embedding.create('glove', source=args.embedding_source, unknown_token='<unk>',
                                         init_unknown_vec=mx.nd.random_uniform)
    vocab.set_embedding(glove_twitter)
    ctx = mx.cpu()  ## or mx.gpu(N) if GPU device N is available

    train_classifier(vocab, train_dataset, val_dataset, test_dataset, ctx)
