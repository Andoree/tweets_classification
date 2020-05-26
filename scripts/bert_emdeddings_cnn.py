from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout


class TextCNNWithDynamicEmbeddings(Model):

    def __init__(self,
                 maxlen,
                 kernel_sizes=(3, 4, 5),
                 class_num=1,
                 last_activation='sigmoid',):
        super(TextCNNWithDynamicEmbeddings, self).__init__()
        self.maxlen = maxlen
        self.kernel_sizes = kernel_sizes
        self.class_num = class_num
        self.last_activation = last_activation
        #self.input_shape = (-1, maxlen, self.embeddings_dim)
        # self.embedding = Embedding(self.max_features, self.embedding_dims,
        #                            input_length=self.maxlen, weights=[embedding_weights], )
        self.convs = []
        self.max_poolings = []
        for kernel_size in self.kernel_sizes:
            self.convs.append(Conv1D(128, kernel_size, activation='relu'))
            self.max_poolings.append(GlobalMaxPooling1D())
        self.dropout = Dropout(0.5)
        self.classifier = Dense(self.class_num, activation=self.last_activation)

    def call(self, inputs):
        #if len(inputs.get_shape()) != 2:
        #    raise ValueError('The rank of inputs of TextCNN must be 2, but now is %d' % len(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of TextCNN must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        convs = []
        for i in range(len(self.kernel_sizes)):
            c = self.convs[i](inputs)
            c = self.max_poolings[i](c)
            convs.append(c)
        x = Concatenate()(convs)
        x = self.dropout(x)
        output = self.classifier(x)
        return output
