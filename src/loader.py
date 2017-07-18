import os
import re
import codecs
from utils import create_dico, create_mapping, zero_digits
from utils import iob2, iob_iobes
unknown_word='<UNK>';

def load_sentences(path, lower, zeros):
#{{{
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences
#}}}

def update_tag_scheme(sentences, tag_scheme,removeTag=None):
#{{{
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                if removeTag is not None:
                    if new_tag[2:] in removeTag:
                        word[-1]='O';
                    else:
                        word[-1]=new_tag;
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')
#}}}

def word_mapping(sentences, lower):
#{{{
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(words)
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    print "Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in words)
    )
    return dico, word_to_id, id_to_word
#}}}

def feature_mapping(sentences,index,featureName="",isPos=False):
#{{{
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    if isPos:
        features = [[w[0].lower()+"_"+w[index] for w in s] for s in sentences]
    else: 
        features = [[w[index] for w in s] for s in sentences]
    dico = create_dico(features)
    dico[unknown_word]=10000000
    feature_to_id, id_to_feature = create_mapping(dico)
    print "Found %i unique %s features" % (len(dico),featureName)
    return dico, feature_to_id, id_to_feature
#}}}

def char_mapping(sentences):
#{{{
    """
    Create a dictionary and mapping of characters, sorted by frequency.
    """
    chars = ["".join([w[0] for w in s]) for s in sentences]
    dico = create_dico(chars)
    char_to_id, id_to_char = create_mapping(dico)
    print "Found %i unique characters" % len(dico)
    return dico, char_to_id, id_to_char
#}}}

def tag_mapping(sentences):
#{{{
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print "Found %i unique named entity tags" % len(dico)
    return dico, tag_to_id, id_to_tag
#}}}

def cap_feature(s):
#{{{
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3
#}}}

def prepare_sentence(str_words, word_to_id, char_to_id, lower=False):
#{{{
    """
    Prepare a sentence for evaluation.
    """
    def f(x,flag=lower): return x.lower() if flag else x
    words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
             for w in str_words]
    charLower=False;
    if charLower:
        chars = [[char_to_id[c] for c in w.lower() if c in char_to_id]
                 for w in str_words]
    else:
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
    caps = [cap_feature(w) for w in str_words]
    return {
        'str_words': str_words,
        'words': words,
        'chars': chars,
        'caps': caps
    }
#}}}

def prepare_dataset(sentences,docLen,parameters,
                        lower=False,isTest=False):
#{{{
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower() if lower else x
    #get mapping 
#{{{
    features=parameters['features'];
    feature2IdMap=features['feature2IdMap'];
    word_to_id=feature2IdMap['word'];
    char_to_id=feature2IdMap['char'];
    tag_to_id=feature2IdMap['tag'];
    if features['lemma']['isUsed']:
        lemma_to_id=feature2IdMap['lemma'];
    if features['pos']['isUsed']:
        pos_to_id=feature2IdMap['pos'];
    if features['chunk']['isUsed']:
        chunk_to_id=feature2IdMap['chunk'];
    if features['dic']['isUsed']:
        dic_to_id=feature2IdMap['dic'];
#}}}
    data = []
    if docLen is not None and len(sentences) != len(docLen):
        print "len(doc) != len(docLen)";
        assert 0;
    i=0;
    for s in sentences:
        str_words = [w[0] for w in s]
        elem=prepare_sentence(str_words,word_to_id,char_to_id,lower);
        words = elem['words']
        # Skip characters that are not in the training set
        chars = elem['chars']
        caps = elem['caps'];
        if not isTest:
            tags = [tag_to_id[w[-1]] for w in s]
        
        e={
           'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
        }
       
        #add features
#{{{
        if features['lemma']['isUsed']:
            lemma=[lemma_to_id[w[1]] 
                        if w[1] in lemma_to_id 
                                else lemma_to_id[unknown_word] for w in s];
            e['lemma']=lemma;
        if features['pos']['isUsed']:
            pos=[pos_to_id[w[2]] 
                        if w[2] in pos_to_id 
                                else pos_to_id[unknown_word] for w in s];
            e['pos']=pos;
        if features['chunk']['isUsed']:
            chunk=[chunk_to_id[w[3]] 
                        if w[3] in chunk_to_id 
                                else chunk_to_id[unknown_word] for w in s];
            e['chunk']=chunk;
        if features['dic']['isUsed']:
            ner=[dic_to_id[w[4]] for w in s];
            e['dic']=ner; 
       #}}}

        #append doc len to data  
        if parameters.has_key('sentencesLevelLoss') \
                and parameters['sentencesLevelLoss']:
            lens=docLen[i];
            i+=1;
            e['lens']=lens;
       
        if not isTest:
            e['tags']=tags;
        


        data.append(e);
    return data
#}}}

def augment_with_pretrained(dictionary, ext_emb_path, words):
#{{{
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print 'Loading pretrained embeddings from %s...' % ext_emb_path
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word 
#}}}
