"""
    InpFile: file descriptor (file object)
    maxlen: the length of sequence
"""

def readFile(inpFile,maxlen):
    _hash={}
    _seq=[]

    for line in open(inpFile):
        if line.startswith('>'):
           name=line.replace('>','').split()[0]
           _hash[name]=''
        else:
           _hash[name]+=line.replace('\n','')

    for i in _hash.keys():
        if len(_hash[i])<= maxlen:
           Proseq=_hash[i] + "X" * (maxlen - len(_hash[i]))
        else:
           Proseq=_hash[i][0:maxlen]
        _seq.append(Proseq)

    return _seq
