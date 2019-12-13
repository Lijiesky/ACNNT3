import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score,confusion_matrix,matthews_corrcoef 
import numpy as np
from fasta_reader import readFile
from helpers import *
# from plot import *
import keras
from keras.layers import LSTM
from sklearn.model_selection import StratifiedKFold
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.constraints import maxnorm

from keras.optimizers import RMSprop, SGD
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
import keras.layers.core as core
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, multiply, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import fbeta_score, roc_curve, auc, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from keras.regularizers import l2, l1, l1_l2
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec
from scipy import interp
from sklearn.model_selection import KFold
# Variable
maxlen = 100
batch_size = 10
epochs = 25
seq_rows, seq_cols =200,20
num_classes = 2
KF = KFold(n_splits = 5)
np.set_printoptions(threshold=np.inf)


tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)

i=1

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

# model = set_up_model_up()



print('Loading training data...')
pos_Train=readFile("./data/pos_training_dataset.txt",maxlen)
neg_Train=readFile("./data/neg_training_dataset_1.txt",maxlen)

print('Generating labels and features...')
# (y_train, x_train)=createTrainData(pos_Train,neg_Train,"Onehot")
(label, data)=createTrainData(pos_Train,neg_Train,"Onehot")

print('Shuffling the data...')
index=np.arange(len(label))
np.random.shuffle(index)
data=data[index,:]
label=label[index]

# x_train = x_train.reshape(x_train.shape[0],seq_rows, seq_cols)
# y_train = keras.utils.to_categorical(y_train, num_classes)

data = data.reshape(data.shape[0],seq_rows, seq_cols)
label = keras.utils.to_categorical(label, num_classes)

print('Training...')
for train_index,val_index in KF.split(data):
	print("第"+str(i)+"重交叉验证")
	#建立模型，并对训练集进行测试，求出预测得分
    #划分训练集和测试集
	x_train,x_val = data[train_index],data[val_index]
	y_train,y_val = label[train_index],label[val_index]
	#建立模型(模型已经定义)
	model = set_up_model_up()
	# model.compile(optimizer = 'sgd',loss = 'categorical_crossentropy',metrics = ['acc'])
	model.fit(x_train,y_train,batch_size = batch_size,validation_data = (x_val,y_val),epochs = epochs,shuffle=False)
	#利用model.predict获取测试集的预测值
	y_pred = model.predict(x_val)
	#计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
	fpr,tpr,thresholds=roc_curve(y_val[:,1],y_pred[:,1])
	#interp:插值 把结果添加到tprs列表中
	tprs.append(interp(mean_fpr,fpr,tpr))
	tprs[-1][0]=0.0
	#计算auc
	roc_auc=auc(fpr,tpr)
	aucs.append(roc_auc)
	#画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
	plt.plot(fpr,tpr,lw=1,alpha=0.3,label='ROC fold %d(area=%0.2f)'% (i,roc_auc))

	model.save_weights('./1-1model/'+'model_' + str(i) + '.h5')

	print('Loading test data...')
	pos_Test = readFile("./data/pos_independent_test_dataset.txt",maxlen)
	neg_Test = readFile("./data/neg_independent_test_dataset.txt",maxlen)

	# pos_Test = readFile("./data/P.syringae_nr_effector.txt",maxlen)
	# neg_Test = readFile("./data/neg_P.syringae_test_dataset.txt",maxlen)

	print('Generating labels and features...')
	(y_test, x_test)=createTestData(pos_Test,neg_Test,"Onehot")
	x_test = x_test.reshape(x_test.shape[0],seq_rows,seq_cols)

	print('Evaluating the model')
	model.load_weights('./1-1model/'+'model_' + str(i) + '.h5')
	predicted_Probability = model.predict(x_test)
	# prediction = model.predict_class(x_test)
	prediction = model.predict(x_test)
	prediction=np.argmax(prediction,axis=1)

	print('Showing the confusion matrix')
	cm=confusion_matrix(y_test,prediction)
	print(cm)
	print("ACC: %f "%accuracy_score(y_test,prediction))
	print("F1: %f "%f1_score(y_test,prediction))
	print("Recall: %f "%recall_score(y_test,prediction))
	print("Pre: %f "%precision_score(y_test,prediction))
	print("MCC: %f "%matthews_corrcoef(y_test,prediction))
	print("AUC: %f "%roc_auc_score(y_test,prediction))
	i+=1

#画对角线
plt.plot([0,1],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)#计算平均AUC值
std_auc=np.std(tprs,axis=0)
plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
std_tpr=np.std(tprs,axis=0)
tprs_upper=np.minimum(mean_tpr+std_tpr,1)
tprs_lower=np.maximum(mean_tpr-std_tpr,0)
plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc='lower right')
plt.show()









