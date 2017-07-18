#!/usr/bin/env python

import os
import time
import codecs
import optparse
import numpy as np
from loader import prepare_dataset;
from utils import create_input, iobes_iob;
from model import Model

optparser = optparse.OptionParser()
optparser.add_option(
    "-m", "--model", default="../models/chemAtten_word_char/",
    help="Model location"
)
optparser.add_option(
    "-i", "--input", default="../data/chemnder_test.txt",
    help="Input file location"
)
optparser.add_option(
    "-o", "--output", default="./chemdner_test.tsv",
    help="Output file location"
)
optparser.add_option(
    "-d", "--delimiter", default="__",
    help="Delimiter to separate words from their tags"
)
opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model)
assert os.path.isfile(opts.input)

# Load existing model
print "Loading model..."
model = Model(model_path=opts.model)

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]
parameters = model.parameters

# Load the model
_, f_eval = model.build4(parameters)
model.reload()

#load test sentence  
def load_sentences(path):
    sentences = []
    for line in codecs.open(path, 'r', 'utf8'):
        sentence =[];
        line = line.rstrip()
        if line:
            word = line.split()
            for elem in word:
                sentence.append([elem]);
            sentences.append(sentence)
    return sentences 

opts.train="../data/chemdner_training.ner.doc.token4.BIO_allfea";
opts.dev="../data/chemdner_development.ner.doc.token4.BIO_allfea";
opts.test="../data/chemdner_evaluation.ner.doc.token4.BIO_allfea";
ssplitTrainFName="../data/training.ner.ssplit.token4.BIO";
ssplitDevFName="../data/development.ner.ssplit.token4.BIO";
ssplitTestFName="../data/evaluation.ner.ssplit.token4.BIO";
from utils import generateDocSentLen;
#get doc Len  for calcuate loss at sentences level
train_Lens=generateDocSentLen(opts.train,ssplitTrainFName);
dev_Lens=generateDocSentLen(opts.dev,ssplitDevFName);
test_Lens=generateDocSentLen(opts.test,ssplitTestFName);

test_sentences=load_sentences(opts.input);
test_data=prepare_dataset(test_sentences,test_Lens,parameters,parameters['lower'],isTest=True);
f_output = codecs.open(opts.output, 'w', 'utf-8')
start = time.time()

def xmlformat(sentence,tags):
#{{{
    assert len(sentence)==len(tags);
    res=[];
    preTag="";
    for i in range(len(tags)):
        if tags[i][0]=='B':
            if len(preTag):
                res.append("</"+preTag+">");
                preTag="";
            res.append("<"+tags[i][2:]+">");
            preTag=tags[i][2:];
        if tags[i][0]=='I':
            if preTag!=tags[i][2:]:
                if len(preTag):
                    res.append("</"+preTag+">");
                    preTag="";

        if tags[i][0]=='O':
            if len(preTag):
                res.append("</"+preTag+">");
                preTag="";
        res.append(sentence[i]);
    if len(preTag):
        res.append("</"+preTag+">");
    return res;
#}}}
print 'Tagging...'
for line in test_data:
    # Prepare input
    input = create_input(line, parameters, False,useAttend=parameters['useAttend']);
    words=line['str_words'];
    # Decoding
    if parameters['crf']:
        y_preds = np.array(f_eval(*input))
    else:
        y_preds = f_eval(*input).argmax(axis=1)
    y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]
    # Output tags in the IOB2 format
    if parameters['tag_scheme'] == 'iobes':
        y_preds = iobes_iob(y_preds)
    # Write tags
    assert len(y_preds) == len(words)
    for i in range(len(words)):
        f_output.write(words[i]+'\t'+y_preds[i]+'\n')
    f_output.write('\n')
#    for elem in xmlformat(words,y_preds):
#                    f_output.write(elem+" ");
#    f_output.write("\n");

print '---- lines tagged in %.4fs ----' % ( time.time() - start)
f_output.close()
