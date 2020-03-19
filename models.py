# coding: utf-8

from mxnet import gluon
from mxnet.gluon import HybridBlock

class CNNTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, dropout, lstm, filters=[3,4], num_conv_layers=2,
                 num_classes=2, prefix=None, params=None):
        super(CNNTextClassifier, self).__init__(prefix=prefix, params=params)

        if not lstm:
            with self.name_scope():
                self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
                self.encoder = gluon.nn.HybridSequential()
                with self.encoder.name_scope():
                    self.encoder.add(gluon.nn.HybridLambda(lambda F, x: F.expand_dims(x, axis=3)))
                    for i in range(num_conv_layers):
                        self.encoder.add(gluon.nn.Conv2D(layout='NCHW', channels=50, kernel_size=(filters[i],1),strides=(3,1),
                                                     activation='sigmoid'))
                    self.encoder.add(gluon.nn.Dense(64))
                    self.encoder.add(gluon.nn.Dropout(dropout))
                self.output = gluon.nn.HybridSequential()
                with self.output.name_scope():
                    self.output.add(gluon.nn.Dense(num_classes))
        else:
            with self.name_scope():
                self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
                self.encoder = gluon.nn.HybridSequential()
                with self.encoder.name_scope():
                    self.encoder.add(gluon.nn.Conv1D(channels=50,kernel_size=filters[1],strides=3))
                    self.encoder.add(gluon.nn.HybridLambda(lambda F, x: F.swapaxes(x, 1,2)))
                    self.encoder.add(gluon.rnn.LSTM(hidden_size=32,dropout=0.5,layout='NTC'))
                    self.encoder.add(gluon.nn.Dense(12, activation='sigmoid'))
                    self.encoder.add(gluon.nn.Dropout(dropout))
                self.output = gluon.nn.HybridSequential()
                with self.output.name_scope():
                    self.output.add(gluon.nn.Dense(num_classes))

    def hybrid_forward(self, F, data):
        embedded = self.embedding(data)
        encoded = self.encoder(embedded)
        return self.output(encoded)

class DANTextClassifier(HybridBlock):

    def __init__(self, emb_input_dim, emb_output_dim, dropout, num_classes=2, prefix=None, params=None):
        super(DANTextClassifier, self).__init__(prefix=prefix, params=params)
        
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(emb_input_dim, emb_output_dim)
            self.encoder = gluon.nn.HybridSequential()
            self.pooler = gluon.nn.GlobalAvgPool1D()
            with self.encoder.name_scope():
                self.encoder.add(gluon.nn.Dense(128, activation='relu'))
                self.encoder.add(gluon.nn.Dropout(0.5))
                self.encoder.add(gluon.nn.Dense(64, activation='relu'))
                self.encoder.add(gluon.nn.Dropout(0.5))
                self.encoder.add(gluon.nn.Dense(12, activation='relu'))
                self.encoder.add(gluon.nn.Dropout(0.5))
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dense(num_classes))

    def hybrid_forward(self, F, data):
        embedded = self.embedding(data)
        pooled = self.pooler(embedded)
        encoded = self.encoder(pooled)
        return self.output(encoded)
            
    
