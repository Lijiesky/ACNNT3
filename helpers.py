from fasta_reader import readFile
import numpy as np
import glob as glob
def OnehotEncoding(inpStr):
    _res = []
    for base in inpStr:
        if base == "G":
            base = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "A":
            base = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "V":
            base = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "L":
            base = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "I":
            base = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "P":
            base = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "F":
            base = [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "Y":
            base = [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "W":
            base = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
        elif base == "S":
            base = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
        elif base == "T":
            base = [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        elif base == "C":
            base = [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]
        elif base == "M":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif base == "N":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
        elif base == "Q":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]
        elif base == "D":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]
        elif base == "E":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
        elif base == "K":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
        elif base == "R":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]
        elif base == "H":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif base == "X":
            base = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        _res.append(base)

    return _res


def dictEncoding(inpStr):
    _res = []
    word2int = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}

    for inp in range(len(inpStr)):
        _res.append(word2int.get(inpStr[inp]))

    return _res


def createTrainData(posSample,negSample,Encodingtype):
    TrainTest=[]
    seq_len=[]
    num=[]
    pos_label = np.ones((len(posSample),1))
    neg_label = np.zeros((len(negSample),1))
    Label = np.concatenate((pos_label,neg_label),axis=0).flatten()
    TrainTestSample = posSample + negSample

    if Encodingtype == "dict":
       for i in TrainTestSample:
           seq_len=len(i)
           i=np.array(dictEncoding(i)).reshape([1,seq_len])
           TrainTest.append(i)
       TrainTest=np.array(TrainTest).reshape(len(TrainTestSample),seq_len)
       return Label, TrainTest
    else:
       for i in TrainTestSample:
           num = len(i) * 20
           i=np.array(OnehotEncoding(i)).reshape([1,num])
           TrainTest.append(i)

       TrainTest=np.array(TrainTest).reshape(len(TrainTestSample),2000)
       TrainTest=np.concatenate([TrainTest, train3()], axis=1)
       # TrainTest=np.array(TrainTest).reshape(len(TrainTestSample),4000)
       # print(np.shape(TrainTest))
       return Label, TrainTest
def createTestData(posSample,negSample,Encodingtype):
    TrainTest=[]
    seq_len=[]
    num=[]
    pos_label = np.ones((len(posSample),1))
    neg_label = np.zeros((len(negSample),1))
    Label = np.concatenate((pos_label,neg_label),axis=0).flatten()
    TrainTestSample = posSample + negSample

    if Encodingtype == "dict":
       for i in TrainTestSample:
           seq_len=len(i)
           i=np.array(dictEncoding(i)).reshape([1,seq_len])
           TrainTest.append(i)
       TrainTest=np.array(TrainTest).reshape(len(TrainTestSample),seq_len)
       return Label, TrainTest
    else:
       for i in TrainTestSample:
           num = len(i) * 20
           i=np.array(OnehotEncoding(i)).reshape([1,num])
           TrainTest.append(i)

       TrainTest=np.array(TrainTest).reshape(len(TrainTestSample),2000)
       TrainTest=np.concatenate([TrainTest, test3()], axis=1)

       print(np.shape(TrainTest))
       return Label, TrainTest




def simplifypssm(pssmdir):
    TrainTest=[]
    filelist = glob.glob(pssmdir+'/*.pssm')
    file_num = len(filelist)
    for i in range(file_num):
        file = pssmdir+'/seq'+str(i+1)+'.pssm'
        with open(file,'r') as inputpssm:
            aa = []
            filelist = inputpssm.readlines()
            for line in filelist:
                returnMat = np.zeros((20))
                line = line.strip()
                listline=line.split('\t')
                returnMat[:20] =listline[0:20]
                aa.append(returnMat)
            mm=np.array(aa).reshape(1,2000)
            TrainTest.append(mm)
            del aa,mm
    return TrainTest

def train3():
    # ACNNT3-1
    pssmdirtrain1 = './pssm/pos_training_dataset'
    pssmdirtrain2 = './pssm/neg_training_dataset_1'

    # ACNNT3-2
    # pssmdirtrain1 = './pssm/pos_training_dataset'
    # pssmdirtrain2 = './pssm/neg_training_dataset_2'

    # pos_training_dataset:283个  neg_training_dataset_1:311个  neg_training_dataset_2:835个

    train1=simplifypssm(pssmdirtrain1)
    train1=np.array(train1).reshape(283,2000)


    train2=simplifypssm(pssmdirtrain2)
    train2=np.array(train2).reshape(311,2000)

    train3=np.concatenate([train1, train2], axis=0)
    print(np.shape(train3))
    return train3

def test3():
    # Independent dataset
    pssmdirtrain1 = './pssm/pos_independent_test_dataset'
    pssmdirtrain2 = './pssm/neg_independent_test_dataset'

    # P.syringae dataset
    # pssmdirtrain1 = './pssm/P.syringae_nr_effector'
    # pssmdirtrain2 = './pssm/neg_P.syringae_test_dataset'


    # pos_independent_test_dataset:35个  neg_independent_test_dataset:86个
    # P.syringae_nr_effector:83个  neg_P.syringae_test_dataset:14个

    test1=simplifypssm(pssmdirtrain1)
    test1=np.array(test1).reshape(35,2000)

    test2=simplifypssm(pssmdirtrain2)
    test2=np.array(test2).reshape(86,2000)

    test3=np.concatenate([test1, test2], axis=0)
    return test3

















