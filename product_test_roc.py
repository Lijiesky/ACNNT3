from helpers import *
from plot import *
from keras.layers.core import Activation, Flatten
from keras.layers import Dense, Dropout, Input, merge
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.constraints import maxnorm
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints

maxlen=100
seq_rows,seq_cols=200,20

#build Attention model
class Attention(Layer):
	def __init__(self,hidden,init='glorot_uniform',activation='linear',W_regularizer=None,b_regularizer=None,W_constraint=None,**kwargs):
	    self.init = initializers.get(init)
	    self.activation = activations.get(activation)
	    self.W_regularizer = regularizers.get(W_regularizer)
	    self.b_regularizer = regularizers.get(b_regularizer)
	    self.W_constraint = constraints.get(W_constraint)
	    self.hidden=hidden
	    super(Attention, self).__init__(**kwargs)

	def build(self, input_shape):
	    input_dim = input_shape[-1]
	    self.input_length = input_shape[1]
	    self.W0 = self.add_weight(name ='{}_W1'.format(self.name), shape = (input_dim, self.hidden), initializer = 'glorot_uniform', trainable=True) # Keras 2 API
	    self.W  = self.add_weight( name ='{}_W'.format(self.name),  shape = (self.hidden, 1), initializer = 'glorot_uniform', trainable=True)
	    self.b0 = K.zeros((self.hidden,), name='{}_b0'.format(self.name))
	    self.b  = K.zeros((1,), name='{}_b'.format(self.name))
	    self.trainable_weights = [self.W0,self.W,self.b,self.b0]

	    self.regularizers = []
	    if self.W_regularizer:
	        self.W_regularizer.set_param(self.W)
	        self.regularizers.append(self.W_regularizer)

	    if self.b_regularizer:
	        self.b_regularizer.set_param(self.b)
	        self.regularizers.append(self.b_regularizer)

	    self.constraints = {}
	    if self.W_constraint:
	        self.constraints[self.W0] = self.W_constraint
	        self.constraints[self.W] = self.W_constraint

	    super(Attention, self).build(input_shape)

	def call(self,x,mask=None):
	        attmap = self.activation(K.dot(x, self.W0)+self.b0)
	        attmap = K.dot(attmap, self.W) + self.b
	        attmap = K.reshape(attmap, (-1, self.input_length)) # Softmax needs one dimension
	        attmap = K.softmax(attmap)
	        dense_representation = K.batch_dot(attmap, x, axes=(1, 1))
	        out = K.concatenate([dense_representation, attmap]) # Output the attention maps but do not pass it to the next layer by DIY flatten layer
	        return out


	def compute_output_shape(self, input_shape):
	    return (input_shape[0], input_shape[-1] + input_shape[1])

	def get_config(self):
	    config = {'init': 'glorot_uniform',
	              'activation': self.activation.__name__,
	              'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
	              'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
	              'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
	              'hidden': self.hidden if self.hidden else None}
	    base_config = super(Attention, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


class attention_flatten(Layer): # Based on the source code of Keras flatten
	def __init__(self, keep_dim, **kwargs):
	    self.keep_dim = keep_dim
	    super(attention_flatten, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
	    if not all(input_shape[1:]):
	        raise Exception('The shape of the input to "Flatten" '
	                        'is not fully defined '
	                        '(got ' + str(input_shape[1:]) + '. '
	                        'Make sure to pass a complete "input_shape" '
	                        'or "batch_input_shape" argument to the first '
	                        'layer in your model.')
	    return (input_shape[0], self.keep_dim)   # Remove the attention map

	def call(self, x, mask=None):
	    x=x[:,:self.keep_dim]
	    return K.batch_flatten(x)

def set_up_model_up():
	print('building model')

	seq_input_shape = (200,20)
	nb_filter = 64
	filter_length = 6
	# input_shape = (200,20,1)
	attentionhidden = 256

	seq_input = Input(shape = seq_input_shape, name = 'seq_input')
	convul1   = Convolution1D(filters = nb_filter,
                        	  kernel_size = filter_length,
                        	  padding = 'valid',
                        	  activation = 'relu',
                        	  kernel_constraint = maxnorm(3),
                        	  subsample_length = 1)

	pool_ma1 = MaxPooling1D(pool_size = 3)
	dropout1 = Dropout(0.6)
	dropout2 = Dropout(0.3)
	decoder  = Attention(hidden = attentionhidden, activation = 'linear')
	dense1   = Dense(2)
	dense2   = Dense(2)

	output_1 = pool_ma1(convul1(seq_input))
	output_2 = dropout1(output_1)
	att_decoder  = decoder(output_2)
	output_3 = attention_flatten(output_2._keras_shape[2])(att_decoder)

	output_4 =  dense1(dropout2(Flatten()(output_2)))
	all_outp =  merge([output_3, output_4], mode = 'concat')
	output_5 =  dense2(all_outp)
	output_f =  Activation('sigmoid')(output_5)

	model = Model(inputs = seq_input, outputs = output_f)
	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	# print (model.summary())
	return model

model = set_up_model_up()


model.load_weights('./models_weights/ACNNT3-1.h5')
# model.load_weights('./models_weights/ACNNT3-2.h5')

print('Loading test data...')
pos_Test = readFile("./data/pos_independent_test_dataset.txt",maxlen)
neg_Test = readFile("./data/neg_independent_test_dataset.txt",maxlen)

# pos_Test = readFile("./data/P.syringae_nr_effector.txt",maxlen)
# neg_Test = readFile("./data/neg_P.syringae_test_dataset.txt",maxlen)

print('Generating labels and features...')
(y_test, x_test)=createTestData(pos_Test,neg_Test,"Onehot")
x_test = x_test.reshape(x_test.shape[0],seq_rows,seq_cols)



print('Evaluating the model')
predicted_Probability = model.predict(x_test)
# prediction = model.predict_class(x_test)
prediction = model.predict(x_test)
prediction=np.argmax(prediction,axis=1)

print('Plotting the ROC curve...')
plotROC(y_test,predicted_Probability[:,1])









