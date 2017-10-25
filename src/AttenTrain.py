#!/usr/bin/env python

import os
import numpy as np 
SEED=1234;
np.random.seed(1234);
import optparse
import itertools
import time
import subprocess
from collections import OrderedDict
from utils import create_input
import loader

from utils import models_path, evaluate, eval_script, eval_temp,create_mapping;
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained,feature_mapping;
from model import Model 
from utils import generateDocSentLen;
#import random ;
#for bash color 
BASH_RED="\033[0;31m";
BASH_GREEN="\033[0;32m"
BASH_YELLOW="\033[0;33m"
BASH_CYAN="\033[0;36m"
BASH_CLEAR="\033[0m"

#prepare for model 
#{{{
# Read parameters from command line
#{{{
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="training.ner.doc.token4.BIO",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="development.ner.doc.token4.BIO",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="evaluation.ner.doc.token4.BIO",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iob",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="50",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="./word2vec_model/chemdner_pubmed_drug.word2vec_model_token4_d50",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="1",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.001",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-S","--String",default="",
    help="some about this model"
    )
opts = optparser.parse_args()[0]
#}}}


#according corpus to set some parameter for loading file 
CORPUS="chem";
tagFilter=None;
attenScoreFunTotal=['Euclidean','forwardNN','Cosine','Manhatten'];
attenScoreFun=attenScoreFunTotal[0]
if CORPUS == "chem":
#{{{
    opts.train="./chemdner_corpus/chemdner_training.ner.doc.token4.BIO_allfea";
    opts.dev="./chemdner_corpus/chemdner_development.ner.doc.token4.BIO_allfea";
    opts.test="./chemdner_corpus/chemdner_evaluation.ner.doc.token4.BIO_allfea";
    opts.pre_emb="./word2vec_model/chemdner_pubmed_drug.word2vec_model_token4_d50";
    ssplitTrainFName="./chemdner_corpus/training.ner.ssplit.token4.BIO";
    ssplitDevFName="./chemdner_corpus/development.ner.ssplit.token4.BIO";
    ssplitTestFName="./chemdner_corpus/evaluation.ner.ssplit.token4.BIO";
    tagFilter=None;
#}}}
elif CORPUS == "CDR":
#{{{
    opts.train="./cdr_corpus/cdr_training.ner.doc.token4.BIO_allfea_drug";
    opts.dev="./chemdner_corpus/cdr_development.ner.doc.token4.BIO_allfea_drug";
    opts.test="./chemdner_corpus/cdr_test.ner.doc.token4.BIO_allfea_drug";
    opts.pre_emb="./word2vec_model/chemdner_pubmed_drug.word2vec_model_token4_d50";
    ssplitTrainFName="./chemdner_corpus/cdr_training.ner.sen.token4.BIO_allfea_drug";
    ssplitDevFName="./chemdner_corpus/cdr_development.ner.sen.token4.BIO_allfea_drug";
    ssplitTestFName="./chemdner_corpus/cdr_dtest.ner.sen.token4.BIO_allfea_drug";
    tagFilter=['Disease'];
#}}}

else:
    assert 0,"unknown corpus";

#read word_dim from word2vec_model
#{{{
with open(opts.pre_emb) as file:
    first_line = file.readline()
    #create vec_table
    frequency = int(first_line.split()[0]);
    vec_size = int(first_line.split()[1]);
    opts.word_dim=vec_size;
    opts.word_lstm_dim=vec_size;
#}}}

# Parse parameters 
#{{{
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
#}}}

# Check parameters validity
#{{{
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])
#}}}
# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)
#}}}
#prepare for train 
#{{{

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

#check 1 word sentences 
def check1word(sentences):
    Lens=[];
    for elem in sentences:
        Lens.append(len(elem));
    if min(Lens)==1:
        assert 0;
#check1word(train_sentences);
#check1word(dev_sentences);
#check1word(test_sentences);

#get doc Len  for calcuate loss at sentences level
train_Lens=generateDocSentLen(opts.train,ssplitTrainFName);
dev_Lens=generateDocSentLen(opts.dev,ssplitDevFName);
test_Lens=generateDocSentLen(opts.test,ssplitTestFName);

#merge dev to train 
totalSentences=train_sentences+dev_sentences;
totalLens=train_Lens+dev_Lens;
#redefine train and dev 
#corpus are already random genergated, so no need to shuffly
#random.seed(SEED);
#random.shuffle(totalSentences);
#random.seed(SEED);
#random.shuffle(totalLens);
devRatio=0.1;
devBoundary=int(len(totalSentences)*(1-devRatio))
train_sentences=totalSentences[:devBoundary];
train_Lens=totalLens[:devBoundary];
dev_sentences=totalSentences[devBoundary:];
dev_Lens=totalLens[devBoundary:];

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme,tagFilter);
update_tag_scheme(dev_sentences, tag_scheme,tagFilter);
update_tag_scheme(test_sentences, tag_scheme,tagFilter);

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

#feature mapping 
#{{{
featureMap={#{{{
            'word':{
                        'index':1,
                        'isUsed':1,
                        'lstm-input':1,
                        'attended':1,
            },
            'char':{    
                        'index':0,
                        'isUsed':0,
                        'lstm-input':1,
                        'attended':1,
            },
            'lemma':{   'index':1,
                        'isUsed':0,
                        'num':0,
                        'dim':25,
                        'lstm-input':0,
                        'attended':0,
                        'pre_emb':''},
            'pos':{     'index':2,
                        'isUsed':0,
                        'num':0,
                        'dim':50,
                        'lstm-input':0,
                        'attended':0,
                        'pre_emb':''},
            'chunk':{   'index':3,
                        'isUsed':0,
                        'num':0,
                        'lstm-input':0,
                        'attended':0,
                        'dim':10},
            'dic':{     'index':4,
                        'isUsed':0,
                        'num':3,
                        'lstm-input':0,
                        'attended':0,
                        'dim':5},
           }#}}}
def featureMapCheck(featureMap):
    for item in featureMap:
        assert (not featureMap[item]['isUsed']) or \
            (featureMap[item]['lstm-input'] or featureMap[item]['attended'])
feature2IdMap={'word':word_to_id,
                   'char':char_to_id,
                   'tag':tag_to_id};
featureMapCheck(featureMap);
if featureMap['lemma']['isUsed'] :
    dico_lemma,lemma_to_id,id_to_lemma=feature_mapping(train_sentences,
                                            featureMap['lemma']['index'],'lemma');
    featureMap['lemma']['num']=len(dico_lemma)
    feature2IdMap['lemma']=lemma_to_id;

if featureMap['pos']['isUsed'] :
    dico_pos,pos_to_id,id_to_pos=feature_mapping(train_sentences,
                                            featureMap['pos']['index'],'pos');
    featureMap['pos']['num']=len(dico_pos)
    feature2IdMap['pos']=pos_to_id;
if featureMap['chunk']['isUsed']:
    dico_chunk,chunk_to_id,id_to_chunk=feature_mapping(train_sentences,
                                            featureMap['chunk']['index'],'chunk');
    featureMap['chunk']['num']=len(dico_chunk)
    feature2IdMap['chunk']=chunk_to_id;
    
if featureMap['dic']['isUsed']:
    dico_NER={'B':0,'I':1,'O':2};
    NER_to_id,id_to_NER=create_mapping(dico_NER);
    feature2IdMap['dic']=NER_to_id;
print BASH_YELLOW+str(featureMap)+BASH_CLEAR;
featureMap['feature2IdMap']=feature2IdMap;
parameters['features']=featureMap;
#}}}



# Build the model 
parameters['loading']=False;
parameters['loading_path']="./models/bilstm-crf-chemdner50d/";
parameters['sentencesLevelLoss']=False;
saveModel=False;
parameters['training']=True;
parameters['attenScoreFun']=attenScoreFun;
parameters['useAttend']=True;
useEarlyStopping=False;
# Initialize model
model = Model(parameters=parameters, models_path=models_path,model_path="./models/attention_test/",Training=True)
# Save the mappings to disk
print 'Saving the mappings to disk...'
model.save_mappings(id_to_word, id_to_char, id_to_tag)
print BASH_YELLOW+"Model location: "+BASH_CLEAR+ "%s" % model.model_path
print BASH_YELLOW+"model important point:"+BASH_CLEAR,opts.String;
if parameters['loading']:
    print BASH_YELLOW+"loading:"+BASH_CLEAR,parameters['loading_path'];
print BASH_YELLOW+'save model:'+BASH_CLEAR,saveModel;
print BASH_YELLOW+"sentences Level Loss:"+BASH_CLEAR,parameters['sentencesLevelLoss'];

# Index data
train_data = prepare_dataset(
    train_sentences,train_Lens, parameters, lower
)
dev_data = prepare_dataset(
    dev_sentences,dev_Lens,parameters, lower
)
test_data = prepare_dataset(
    test_sentences,test_Lens, parameters, lower
)

print "%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data))

#load pre-train word_embending 
f_train, f_eval = model.build4(parameters)


# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    model.reload()
#}}}
#
# Train network
#
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
freq_eval = int(len(train_data)*0.3)  # evaluate on dev every freq_eval steps
count = 0
limitPrint=0;
param = {
         #'lr':0.005,
         'lr':0.001,
         'verbose':1,
         'decay':True, # decay on the learning rate if improvement stops
         'bs':5, # number of backprop through time steps
         'seed':345,
         'epochs':30,
         'crf':True,
         'shuffle':True};
folder_out = '../log/Attention/'
print BASH_YELLOW+"folder_out:"+BASH_CLEAR,folder_out;
best_f1=-np.inf;

def attenVisualFun(words,energy,index):
#{{{
    print "energy should:",energy[index][index],words[index];
    print "filter energy:";
    energyInd=energy[index].argsort()[::-1][:10];
    attenVisual=[];
    for i in energyInd:
        attenVisual.append([words[i],energy[index][i]]);
    print attenVisual;
    
    #print energyInd;
    #for i in range(len(words)):
    #    attenVisual.append([words[i],energy[0][i]]);
    #print attenVisual;
    
    return ;
#}}}

#generate FILE NAME PREFIX 
fileNamePrefix="";
if opts.String != "":
    fileNamePrefix=opts.String;
    fileNamePrefix.replace(",","_");
    fileNamePrefix.replace(" ","_");
#train model 
if useEarlyStopping:
#{{{
    from utils import EarlyStopping;
    eStop=EarlyStopping(mode='max');
    eStop.on_train_begin();
    
    #start train our model
    for epoch in xrange(param['epochs']):
        epoch_costs = []
        startTime=time.time();
        
        #decide whether early stop 
        if eStop.stop_training:
            break;
        
        print "Starting epoch %i..." % epoch
        for i, index in enumerate(np.random.permutation(len(train_data))):
            count += 1
            input = create_input(train_data[index], parameters, True, singletons)
            new_cost = f_train(*input)
            if np.isnan(new_cost):
                print index,"nan"
            epoch_costs.append(new_cost)
        #validation
        res_dev = evaluate(parameters, f_eval, dev_sentences,
                              dev_data, id_to_tag, dico_tags,
                            folder_out+fileNamePrefix+'.dev.txt')
        eStop.on_epoch_end(epoch,res_dev['f1']) ;
        print BASH_YELLOW+"avg error:"+BASH_CLEAR,np.mean(epoch_costs),\
                    " dev F1:",res_dev['f1'];
        print BASH_YELLOW+"One epch espliced:"+BASH_CLEAR,time.time()-startTime;

    #start evaluate on test
    res_test = evaluate(parameters, f_eval, test_sentences,
                      test_data, id_to_tag, dico_tags,
                    folder_out+fileNamePrefix+'.test.txt')
    if saveModel:
        print "Saving model to disk..."
        model.save()
    print BASH_RED+'TEST: epoch'+BASH_CLEAR, epoch, 'F1', res_test['f1'],'p:',res_test['p'],'r:',res_test['r'],  ' '*15
    print BASH_YELLOW+"model important point:"+BASH_CLEAR,opts.String;
            #}}}
else:
    for epoch in xrange(param['epochs']):
        epoch_costs = []
        startTime=time.time();
        print "Starting epoch %i..." % epoch
        for i, index in enumerate(np.random.permutation(len(train_data))):
            count += 1
            input = create_input(train_data[index], parameters, True, singletons)
            new_cost,energy = f_train(*input)
            #print attention energy for test
            if epoch>=limitPrint and count %freq_eval==0:
                attenVisualFun(train_data[index]['str_words'],
                              energy,
                               np.random.randint(0,len(train_data[index])));
            if np.isnan(new_cost):
                print "NaN,index:",index;
            epoch_costs.append(new_cost)
            if count % freq_eval == 0 and epoch>=limitPrint:
                res_dev = evaluate(parameters, f_eval, dev_sentences,
                                      dev_data, id_to_tag, dico_tags,
                                    folder_out+fileNamePrefix+'.dev.txt')
                #new F1 value on dev 
                if res_dev['f1'] > best_f1:
                    best_f1 = res_dev['f1']
                    if param['verbose']:
                        print BASH_CYAN+'NEW DEV BEST: epoch'+BASH_CLEAR, epoch, 'best dev F1', res_dev['f1'],'p:',res_dev['p'],'r:',res_dev['r'],  ' '*15 
                    
                    #new F1 value on dev, so evaluate on test
                    res_test = evaluate(parameters, f_eval, test_sentences,
                                      test_data, id_to_tag, dico_tags,
                                    folder_out+fileNamePrefix+'.test.txt')
                    if saveModel:
                        print "Saving model to disk..."
                        model.save()
                    print BASH_RED+'THIS TEST: epoch'+BASH_CLEAR, epoch, 'F1', res_test['f1'],'p:',res_test['p'],'r:',res_test['r'],  ' '*15
                    param['tf1'], param['tp'], param['tr'] = res_test['f1'],  res_test['p'],  res_test['r']
                    param['be'] = epoch
        print BASH_YELLOW+"avg error:"+BASH_CLEAR,np.mean(epoch_costs);
        print BASH_YELLOW+"One epch espliced:"+BASH_CLEAR,time.time()-startTime;
    print BASH_GREEN+'FINAL TEST RESULT: epoch'+BASH_CLEAR, param['be'], 'final test F1', param['tf1'],'best p:',param['tp'],'best r:',param['tr'] 
    print BASH_YELLOW+"model important point:"+BASH_CLEAR,opts.String;
                

