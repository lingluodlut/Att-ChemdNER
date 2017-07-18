import os
import re
import codecs
import numpy as np 
import six
import theano


models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")

class EarlyStopping(object):
#{{{
    '''Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
    '''
    def __init__(self, monitor='val_loss', 
                 min_delta=1e-6, patience=5,mode='min'):
#{{{
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training=False;
        
        if mode =="min":
            self.monitor_op = np.less;
        elif mode == "max":
            self.monitor_op = np.greater;
        else:
            assert 0,"unknown early stop mode:";

        self.min_delta *= -1
#}}}
    def on_train_begin(self):
        self.wait = 0       # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, loss):
#{{{
        current = loss

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
            self.wait += 1
#}}}
    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0 :
            print('Epoch %05d: early stopping' % (self.stopped_epoch))

#}}}
def get_from_module(identifier, module_params, module_name,
                    instantiate=False, kwargs=None):
    #{{{
    if isinstance(identifier, six.string_types):
        res = module_params.get(identifier)
        if not res:
            raise ValueError('Invalid ' + str(module_name) + ': ' +
                             str(identifier))
        if instantiate and not kwargs:
            return res()
        elif instantiate and kwargs:
            return res(**kwargs)
        else:
            return res
    elif isinstance(identifier, dict):
        name = identifier.pop('name')
        res = module_params.get(name)
        if res:
            return res(**identifier)
        else:
            raise ValueError('Invalid ' + str(module_name) + ': ' +
                             str(identifier))
    return identifier
#}}}

def findNotSame(fNameX,fNameY):
#{{{
    """
    verify two file is same or not 
    """
    space='space';
    def loadFile(fName):
        word=[];
        import codecs;
        for line in codecs.open(fName,'r','utf8'):
            line=line.rstrip();
            if len(line)>0:
                word.append(line[0]);
            else:
                word.append(space);
        return word;
    word1=loadFile(fNameX);
    word2=loadFile(fNameY);
    i=0;
    j=0;
    while i<len(word1) and j<len(word2):
        if word1[i]==word2[j]:
            i+=1;
            j+=1;
            continue;
        elif word1[i] ==space:
            i+=1;
        elif word2[j]==space:
            j+=1;
        else:
            print "not same,X:",word1[i],",line:",i,',Y:',word2[j],',line:',j;
            break;
#}}}

def generateDocSentLen(fNameX,fNameY):
#{{{
    """
    statistic one article have word in each sentence
    """
    from loader import load_sentences;
    doc=load_sentences(fNameX,False,False);
    sent=load_sentences(fNameY,False,False);
    assert len(doc) < len(sent);
    res=[];
    i=0;
    for elem in doc:
        docLen=[];
        count=0;
        while count<len(elem):
            docLen.append(len(sent[i]));
            count+=len(sent[i]);
            i+=1;
        if count!=len(elem):
            print "two file len not same";
            assert 0;
        res.append(docLen)
    
    return res;
#}}}

def get_name(parameters):
#{{{
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    return "".join(i for i in name if i not in "\/:*?<>|")
#}}}

def set_values(name, param, pretrained):
#{{{
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))
#}}}

import initializations;
def shared(shape, name):
#{{{
    """
    Create a shared object of a numpy array.
    """ 
    init=initializations.get('glorot_uniform');
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
        return theano.shared(value=value.astype(theano.config.floatX), name=name)
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
        return init(shape=shape,name=name);
#}}}

def create_dico(item_list):
#{{{
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico
#}}}

def create_mapping(dico):
#{{{
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item
#}}}

def zero_digits(s):
#{{{
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)
#}}}

def iob2(tags):
#{{{
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if split[0] not in ['I', 'B']:
        #if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True
#}}}

def iob_iobes(tags):
#{{{
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags
#}}}

def iobes_iob(tags):
#{{{
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags
#}}}

def insert_singletons(words, singletons, p=0.5):
#{{{
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words
#}}}

def pad_word_chars(words):
#{{{
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos
#}}}


def create_input(data, parameters, add_label, singletons=None,
                useAttend=True):
#{{{
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    wordsTrue=data['words'];
    chars = data['chars']
    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []
    if parameters['word_dim']:
        input.append(words)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)
    if parameters['cap_dim']:
        input.append(caps)
    if useAttend:
        input.append(wordsTrue);
        if parameters.has_key('sentencesLevelLoss') \
                and parameters['sentencesLevelLoss']:
            input.append(data['lens']) ;
    
    #add features 
    if parameters.has_key('features'):
        features=parameters['features'];
    else:
        features=None;
    if features is not None and features['lemma']['isUsed']:
        input.append(data['lemma']);
    if features is not None and features['pos']['isUsed']:
        input.append(data['pos']);
    if features is not None and features['chunk']['isUsed']:
        input.append(data['chunk']);
    if features is not None and features['dic']['isUsed']:
        input.append(data['dic']);

    if add_label:
        input.append(data['tags'])
    return input
#}}}

from os.path import isfile
from os import chmod
import stat
import subprocess
PREFIX = './evaluation/'
def get_perf(filename):
    ''' run conlleval.pl perl script to obtain
    precision/recall and F1 score '''
    _conlleval = PREFIX + 'conlleval'
    if not isfile(_conlleval):
        #download('http://www-etud.iro.umontreal.ca/~mesnilgr/atis/conlleval.pl') 
        os.system('wget https://www.comp.nus.edu.sg/%7Ekanmy/courses/practicalNLP_2008/packages/conlleval.pl')
        chmod('conlleval.pl', stat.S_IRWXU) # give the execute permissions
    
    out = []
    proc = subprocess.Popen(["perl", _conlleval], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename).read())
    for line in stdout.split('\n'):
        if 'accuracy' in line:
            out = line.split()
            break
    
    # out = ['accuracy:', '16.26%;', 'precision:', '0.00%;', 'recall:', '0.00%;', 'FB1:', '0.00']
    precision = float(out[3][:-2])
    recall    = float(out[5][:-2])
    f1score   = float(out[7])

    return {'p':precision, 'r':recall, 'f1':f1score}

def evaluate(parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, dictionary_tags,filename,
             useAttend=True):
#{{{
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)

    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        input = create_input(data, parameters, False,useAttend=useAttend)
        if parameters['crf']:
            y_preds = np.array(f_eval(*input))
        else:
            y_preds = f_eval(*input).argmax(axis=1)
        y_reals = np.array(data['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")
    #write to file 
    with codecs.open(filename, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
    return get_perf(filename) 
#}}}
