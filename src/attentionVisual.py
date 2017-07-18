#!/usr/mZ=utf-8


import numpy as np; 

#plot attention energy graph 
import pylab as P;
import codecs;
from loader import load_sentences;
#energy file 
attendFName="./evaluation/chemAtten_wordCharPosChunkDicAttended.tsv";
lstmFName="./evaluation/chemAtten_wordCharPosChunkDicLstm.tsv";
attendLstmFName="./evaluation/chemAtten_wordCharPosChunkDicAttendedLSTMInput.tsv";
#get str-word 
ssplitFName="/home/BIO/luoling/chemdner/data/corpus/CHEMDNER/feature/all_fea/chemdner_evaluation_analysis.ner.doc.token4.BIO_allfea";
sent=load_sentences(ssplitFName,0,0);
str_words=[]
for d in sent:
    s=[];
    for elem in d:
        s.append(elem[0]);
    str_words.append(s);
attendFile=codecs.open(attendFName,'r','utf8');
attendLstmFile=codecs.open(attendLstmFName,'r','utf8');
docIndex=0;
count=0;
i=0;
with codecs.open(lstmFName,'r','utf8') as f:
    for line in f:
        attendLine=attendFile.readline();
        attendLstmLine=attendLstmFile.readline();
        line=line.rstrip();
        #empyt line skip
        if not line:
            continue;
        line=line.split();
        try:
            tmp=float(line[0]);
        except ValueError:
            count+=1;
            if count==3:
                #new doc 
                docIndex+=1;
                count=0;
                i=0;
            continue;
        else:
            #this is energy;
            ax=P.subplot(111);
            width=30.;
            i+=1;
            str_word=str_words[docIndex-1];
            assert len(str_word)==len(line);

            #lstm energy
            arr=np.array(line).astype(float);
            graph1=ax.bar((np.arange(len(arr))+1)*4*width-width,arr,width=width,color='b',align='center');
            #attended energy 
            attendLine=attendLine.rstrip();
            attendLine=attendLine.split();
            arr=np.array(attendLine).astype(float);
            graph2=ax.bar((np.arange(len(arr))+1)*4*width,arr,width=width,color='g',align='center');
            #attended lstm energy
            attendLstmLine=attendLstmLine.rstrip();
            attendLstmLine=attendLstmLine.split();
            arr=np.array(attendLstmLine).astype(float);
            graph3=ax.bar((np.arange(len(arr))+1)*4*width+width,arr,width=width,color='r',align='center');
            
            #set legend
            P.title("current word: "+str_word[i-1]);
            P.legend((graph1[0],graph2[0],graph3[0]),('lstm','attended','all'));
            P.xticks((np.arange(len(arr))+1)*4*width,str_word,fontsize=10);
            P.show();

            #save to file 
            #figure=P.gcf();
            #figure.set_size_inches(10,10);
            #P.savefig("pdf/"+str(docIndex)+'-'+str(i)+'.pdf')#,dpi=900);
            #exit(0);

attendLstmFile.close();
attendFile.close();

"""
    import numpy as np; 
    np.random.seed(12344)
    from theano import tensor as T;
    import theano;
    from keras import backend as K
    from keras.layers import Embedding;
    import numpy as np 
    from nn import AttentionLayer

    def step(state,attended,source,W_A):
        M=((attended-state)**2).sum(axis=-1)
        _energy=M.max()-M;
        _energy=T.dot(_energy,W_A.T)#+self.b_A;
        energy=T.nnet.softmax(_energy);
        #energyIndex=energy.flatten().argmin(axis=-1);
        glimpsed=(energy.T*source).sum(axis=0)
        #glimpsed=source[energyIndex];
        return energy.flatten(),glimpsed;

    input_dim=5;
    layer=AttentionLayer(attended_dim=input_dim,
                         state_dim=input_dim,
                         source_dim=input_dim);
    state=T.fvector("state");
    attended=T.fmatrix('attended');
    source=T.fmatrix('source');
    y=layer.step(state,attended,source);
    f=theano.function([state,attended,source],y,on_unused_input='ignore')

    state=np.random.random([input_dim,]).astype(theano.config.floatX)
    attended=np.random.random([input_dim,input_dim]).astype(theano.config.floatX)
    source=np.random.random([input_dim,input_dim]).astype(theano.config.floatX)
    print f(state,attended,source)

    x=T.fscalar();
    y=T.sigmoid(x);
    g=theano.function([x],y);
    print g(0.7);

"""


"""
    class ConvLikeEmbedding(Embedding):
        def __init__(self, mergeLen, **kwargs):
            self.mergeLen = mergeLen
            super(ConvLikeEmbedding, self).__init__(**kwargs)

        def build(self, input_shape):
            # Create a trainable weight variable for this layer.
            self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
                                     initializer='random_uniform',
                                     trainable=True)
            super(MyLayer, self).build()  # Be sure to call this somewhere!

        def call(self, x, mask=None):
            return K.dot(x, self.W)

        def get_output_shape_for(self, input_shape):
            return (input_shape[0], self.output_dim)

"""
