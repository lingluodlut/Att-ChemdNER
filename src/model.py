import os
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import cPickle

from utils import shared, set_values, get_name
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, forward 
from nn import LSTM;
#from nn import LSTM_normal as LSTM;
from nn import AttentionLayer;
from optimization import Optimization 

def loadPreEmbFeatures(fName,feature_to_id,weights,lower=False):
#{{{
    def f(x): return x.lower() if lower else x 
    #to lower
    feature_to_id_=feature_to_id;
    if lower:
        feature_to_id_lower={};
        for elem in feature_to_id.items():
            feature_to_id_lower[elem[0].lower()]=elem[1];
        feature_to_id_=feature_to_id_lower;
    feature_dim=weights.shape[1];

    invalid_count=0;
    valid_count=0;
    for line in codecs.open(fName,'r','utf-8'):
        line=line.rstrip().split();
        if len(line) == feature_dim+1 and line[0] in feature_to_id_: 
            weights[feature_to_id_[line[0]]]=np.array(
                [float(x) for x in line[1:]]
                ).astype(theano.config.floatX)
            valid_count+=1;
        else:
            invalid_count+=1;
    print "when loading %s ,%d Invalid line,%d valid line" %(fName,invalid_count,valid_count);
#}}}

class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, 
                 model_path=None,Training=False):
#{{{
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if Training: 
#{{{
            assert parameters and models_path 
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            # Model location 
            if model_path is None:
                model_path = os.path.join(models_path, self.name)
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                cPickle.dump(parameters, f) 
#}}}
        else: 
#{{{
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters=cPickle.load(f);
            self.reload_mappings();
        self.components = {}
#}}}
#}}}
    
    def save_mappings(self, id_to_word, id_to_char, id_to_tag):
#{{{
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
            }
            cPickle.dump(mappings, f)
#}}}

    def reload_mappings(self):
#{{{
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = cPickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']
#}}}

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def modelScore(self,tag_ids,scores,s_len):
    #{{{
        """
            ATTENTATION THIS FUNCTION IS SYMBOL PROGRAMMING
            this function is to return the score of our model at a fixed sentence label 
        @param:
            scores:        the scores matrix ,the output of our model
            tag:           a numpy array, which represent one sentence label 
            sent_lens:     a scalar number, the length of sentence.
                because our sentence label will be expand to max sentence length,
                so we will use this to get the original sentence label. 
        @return: 
            a scalar number ,the score;
        """
    #{{{
        n_tags=self.output_dim;
        transitions=self.transitions;
        #score from tags_scores
        real_path_score = scores[T.arange(s_len), tag_ids].sum()

        # Score from transitions
        b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
        e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))
        padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
        real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()
        #to prevent T.exp(real_path_score) to be inf 
        #return real_path_score;
        return real_path_score/s_len;
    #}}}
    #}}}
   
    def save(self):
#{{{
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)
#}}}

    def reload(self,features=None):
#{{{
        """
        Load components values from disk.
        """
        featureLayerNameMap=['pos_layer','lemma_layer',
                             'chunk_layer','dic_layer'];
        for name, param in self.components.items():
            #when feature is use to attended and not lstm-input, 
            #we will not reload the param
            if features is not None and name in featureLayerNameMap:
                featuresName=name[:name.find('_')];
                if features[featuresName]['attended']==1 and \
                    features[featuresName]['lstm-input']==0:
                    continue;
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])
#}}}
    
    def build4(self,parameters):
        #{{{
        """
        Build the network.
        """
        #some parameters 
        dropout=parameters['dropout'] ;
        char_dim=parameters['char_dim'];
        char_lstm_dim=parameters['char_lstm_dim'];
        char_bidirect=parameters['char_bidirect'];
        word_dim=parameters['word_dim'];
        word_lstm_dim=parameters['word_lstm_dim'];
        word_bidirect=parameters['word_bidirect'];
        lr_method=parameters['lr_method'];
        pre_emb=parameters['pre_emb'];
        crf=parameters['crf'];
        cap_dim=parameters['cap_dim'];
        training=parameters['training'];
        features=parameters['features'];
        useAttend=parameters['useAttend'];
        if useAttend:
            reloadParam=parameters['loading'];
        else:
            reloadParam=None;
        if reloadParam is not None:
            reloadPath=parameters['loading_path']; 
        sentencesLevelLoss=parameters['sentencesLevelLoss'];
        
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)
        self.output_dim = len(self.id_to_tag);
        self.transitions = shared((self.output_dim+ 1, self.output_dim ), 'transitions')

        # Number of capitalization features
        if cap_dim:
            n_cap = 4

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        wordTrue_ids=T.ivector(name='wordTrue_ids');
        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        docLen=T.ivector(name='docLen');
        tag_ids = T.ivector(name='tag_ids')
        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')
        
        #some features
        if features is not None and features['lemma']['isUsed']:
            lemma_ids=T.ivector(name='lemma_ids');
        if features is not None and features['pos']['isUsed']:
            pos_ids=T.ivector(name='pos_ids');
        if features is not None and features['chunk']['isUsed']:
            chunk_ids=T.ivector(name='chunk_ids');
        if features is not None and features['dic']['isUsed']:
            dic_ids=T.ivector(name='dic_ids');

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        # Word inputs
#{{{
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            wordTrue_input=word_layer.link(wordTrue_ids);
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print 'Loading pretrained embeddings from %s...' % pre_emb
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print 'WARNING: %i invalid lines' % emb_invalid
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in xrange(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1
                word_layer.embeddings.set_value(new_weights)
                print 'Loaded %i pretrained embeddings.' % len(pretrained)
                print ('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros, n_words,
                            100. * (c_found + c_lower + c_zeros) / n_words
                      )
                print ('%i found directly, %i after lowercasing, '
                       '%i after lowercasing + zero.') % (
                          c_found, c_lower, c_zeros
                      )#}}}

        # Chars inputs
#{{{
        if char_dim:
            input_dim += char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            char_lstm_for.link(char_layer.link(char_for_ids))
            char_lstm_rev.link(char_layer.link(char_rev_ids))

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_output=T.concatenate([char_for_output,char_rev_output],axis=-1);
            inputs.append(char_for_output)
            if char_bidirect:
                inputs.append(char_rev_output)
                input_dim += char_lstm_dim
#}}}
        
        # Capitalization feature
        #
        if cap_dim:
            input_dim += cap_dim
            cap_layer = EmbeddingLayer(n_cap, cap_dim, name='cap_layer')
            inputs.append(cap_layer.link(cap_ids))
        
        #add feature  
#{{{
        if features is not None and features['lemma']['isUsed']:
            lemma_layer=EmbeddingLayer(features['lemma']['num'],
                                     features['lemma']['dim'],
                                     name='lemma_layer');
            if features['lemma']['pre_emb'] is not "":
                new_weights=lemma_layer.embeddings.get_value();
                loadPreEmbFeatures(features['lemma']['pre_emb'],
                                   features['feature_to_id_map']['lemma'],
                                    new_weights,
                                  lower=True);
                lemma_layer.embeddings.set_value(new_weights); 
            lemma_output=lemma_layer.link(lemma_ids);
            if features['lemma']['lstm-input']:
                input_dim+=features['lemma']['dim'];
                inputs.append(lemma_output);
        if features is not None and features['pos']['isUsed']:
            pos_layer=EmbeddingLayer(features['pos']['num'],
                                     features['pos']['dim'],
                                     name='pos_layer');
            if features['pos']['pre_emb'] is not "":
                new_weights=pos_layer.embeddings.get_value();
                loadPreEmbFeatures(features['pos']['pre_emb'],
                                   features['feature_to_id_map']['pos'],
                                  new_weights);
                pos_layer.embeddings.set_value(new_weights);
            pos_output=pos_layer.link(pos_ids);
            if features['pos']['lstm-input']:
                input_dim+=features['pos']['dim'];
                inputs.append(pos_output);
        if features is not None and features['chunk']['isUsed']:
            chunk_layer=EmbeddingLayer(features['chunk']['num'],
                                     features['chunk']['dim'],
                                     name='chunk_layer');
            chunk_output=chunk_layer.link(chunk_ids);
            if features['chunk']['lstm-input']:
                input_dim+=features['chunk']['dim'];
                inputs.append(chunk_output)
        if features is not None and features['dic']['isUsed']:
            dic_layer=EmbeddingLayer(features['dic']['num'],
                                     features['dic']['dim'],
                                     name='dic_layer');
            dic_output=dic_layer.link(dic_ids);
            if features['dic']['lstm-input']:
                input_dim+=features['dic']['dim'];
                inputs.append(dic_output);
#}}}

        # Prepare final input
        if len(inputs) != 1:
            inputs = T.concatenate(inputs, axis=1)

        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train,input_test);

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev') 
        if sentencesLevelLoss:
            def sentLSTM(i,output,input,lenVec):
    #{{{
                Len=lenVec[i];
                accLen=lenVec[:i].sum();
                currentInput=input[accLen:accLen+Len];
                word_lstm_for.link(currentInput);
                word_lstm_rev.link(currentInput[::-1,:]);
                wordForOutput=word_lstm_for.h;
                wordRevOutput=word_lstm_rev.h[::-1,:];
                finalOutput=T.concatenate(
                        [wordForOutput,wordRevOutput],axis=-1
                        )
                output=T.set_subtensor(output[accLen:accLen+Len],
                                       finalOutput);
                return output;
    #}}}
            result,update=theano.scan(fn=sentLSTM,
                                           outputs_info=T.zeros((inputs.shape[0],word_lstm_dim*2),dtype='float32'),
                                           sequences=[T.arange(docLen.shape[0])],
                                           non_sequences=[inputs,docLen]);
            
            word_lstm_for.link(inputs)
            word_lstm_rev.link(inputs[::-1, :])
            word_for_output = word_lstm_for.h
            word_for_c=word_lstm_for.c;
            word_rev_output = word_lstm_rev.h[::-1, :]
            word_rev_c=word_lstm_rev.c[::-1,:];
            
            final_c=T.concatenate(
                    [word_for_c,word_rev_c],
                    axis=-1
                 )    
            final_output=result[-1]
        else :
            word_lstm_for.link(inputs)
            word_lstm_rev.link(inputs[::-1, :])
            word_for_output = word_lstm_for.h
            word_for_c=word_lstm_for.c;
            word_rev_output = word_lstm_rev.h[::-1, :]
            word_rev_c=word_lstm_rev.c[::-1,:];
            final_output = T.concatenate(
                    [word_for_output, word_rev_output],
                    axis=-1
                )
            final_c=T.concatenate(
                    [word_for_c,word_rev_c],
                    axis=-1
                )
       
        if useAttend:
            #attention layer
            attended=[];
            attendedDim=0;
            if features is not None and features['word']['attended']:
                attended.append(wordTrue_input);
                attendedDim+=word_dim;
            if features is not None and features['char']['attended']:
                attended.append(char_output);
                attendedDim+=char_lstm_dim*2;
            if features is not None and features['lemma']['attended']:
                attended.append(lemma_output);
                attendedDim+=features['lemma']['dim'];
            if features is not None and features['pos']['attended']:
                attended.append(pos_output);
                attendedDim+=features['pos']['dim'];
            if features is not None and features['chunk']['attended']:
                attended.append(chunk_output);
                attendedDim+=features['chunk']['dim'];
            if features is not None and features['dic']['attended']:
                attended.append(dic_output);
                attendedDim+=features['dic']['dim'];
            
            attention_layer=AttentionLayer(attended_dim=attendedDim,
                                           state_dim=attendedDim,
            #attention_layer=AttentionLayer(attended_dim=word_lstm_dim*2,
            #                               state_dim=word_lstm_dim*2,
                                           source_dim=word_lstm_dim*2,
                                           scoreFunName=parameters['attenScoreFun'],
                                          name='attention_layer');

            if len(attended)>1:
                attendedInput=T.concatenate(attended,axis=-1);
            else:
                attendedInput=attended[0];
        
            #final_output=attention_layer.link(attendedInput,attendedInput,final_output);
            #using lstm_state to compute attention
            final_output=attention_layer.link(final_output,final_c,final_output);
            self.energy=attention_layer.energy;
        else:
            final_output=final_output;

        tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                 name='tanh_layer', activation='tanh')
        final_output = tanh_layer.link(final_output)

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:
            if sentencesLevelLoss:
                #calcuate loss according to sentence instead of docLen
                def sentLoss(i,scores,trueIds,transitions,lenVec):
    #{{{
                    Len=lenVec[i];
                    accLen=lenVec[:i].sum();
                    currentTagsScores=scores[accLen:accLen+Len];
                    currentIds=trueIds[accLen:accLen+Len];
                    real_path_score = currentTagsScores[T.arange(Len), 
                                                       currentIds].sum()
                    # Score from transitions
                    padded_tags_ids = T.concatenate([[n_tags],currentIds], axis=0)
                    real_path_score += transitions[
                        padded_tags_ids[T.arange(Len )],
                        padded_tags_ids[T.arange(Len ) + 1]
                    ].sum()

                    all_paths_scores = forward(currentTagsScores,transitions)
                    cost = - (real_path_score - all_paths_scores)
                    return cost;
    #}}}
                result,update=theano.scan(fn=sentLoss,
                                         outputs_info=None,
                                         sequences=[T.arange(docLen.shape[0])],
                                         non_sequences=[tags_scores,tag_ids,self.transitions,docLen])
                cost=result.sum();
            else:
                real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

                # Score from transitions
                padded_tags_ids = T.concatenate([[n_tags], tag_ids], axis=0)
                real_path_score += self.transitions[
                    padded_tags_ids[T.arange(s_len )],
                    padded_tags_ids[T.arange(s_len ) + 1]
                ].sum()

                all_paths_scores = forward(tags_scores, self.transitions)
                cost = - (real_path_score - all_paths_scores)

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            if char_bidirect:
                self.add_component(char_lstm_rev)
                params.extend(char_lstm_rev.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)
        if cap_dim:
            self.add_component(cap_layer)
            params.extend(cap_layer.params)
        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(self.transitions)
            params.append(self.transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)
        #add feature layer
        if features is not None and features['lemma']['isUsed']:
            self.add_component(lemma_layer);
            params.extend(lemma_layer.params);
        if features is not None and features['pos']['isUsed']:
            self.add_component(pos_layer);
            params.extend(pos_layer.params);
        if features is not None and features['chunk']['isUsed']:
            self.add_component(chunk_layer);
            params.extend(chunk_layer.params);
        if features is not None and features['dic']['isUsed']:
            self.add_component(dic_layer);
            params.extend(dic_layer.params);
        
        if useAttend and reloadParam:
            #reload pre-train params 
            model_path=self.model_path;
            self.model_path=reloadPath;
            print "loading:",self.model_path;
            self.reload(features);
            self.model_path=model_path;
        
        if useAttend:
            #add attention_layer
            self.add_component(attention_layer);
            params.extend(attention_layer.params);

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        if cap_dim:
            eval_inputs.append(cap_ids)
        if useAttend:
            eval_inputs.append(wordTrue_ids);
            if sentencesLevelLoss:
                eval_inputs.append(docLen);
        #add feature input 
        if features is not None and features['lemma']['isUsed']:
            eval_inputs.append(lemma_ids);
        if features is not None and features['pos']['isUsed']:
            eval_inputs.append(pos_ids);
        if features is not None and features['chunk']['isUsed']:
            eval_inputs.append(chunk_ids);
        if features is not None and features['dic']['isUsed']:
            eval_inputs.append(dic_ids);
        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print 'Compiling...'
        if training:
            #constraints
            if useAttend:
                self.constraints=attention_layer.constraints;
            else:
                self.constraints={};
            from keras import optimizers ;
            self.optimizer=optimizers.SGD(lr=0.001,momentum=0.9,
                                         decay=0.,nesterov=True,clipvalue=5);
            self.optimizer=optimizers.RMSprop();
            #self.optimizer=SGD(lr=lr_method_parameters['lr'],clipvalue=5,gradient_noise=0.01)
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params,constraints=self.constraints, **lr_method_parameters)
            #updates = self.optimizer.get_updates(params,self.constraints,cost);
            f_train_outputs=[cost];
            if useAttend:
                f_train_outputs.append(self.energy);
 
            f_train = theano.function(
                inputs=train_inputs,
                outputs=f_train_outputs,
                updates=updates,
                on_unused_input='ignore',
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
            
            f_test = theano.function(
                inputs=train_inputs,
                outputs=cost,
                on_unused_input='ignore',
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
            self.f_test=f_test;
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            if sentencesLevelLoss:
                def sentVitebe(i,predictTag,scores,transitions,lenVec):
                    #{{{
                    Len=lenVec[i];
                    accLen=lenVec[:i].sum();
                    currentTagsScores=scores[accLen:accLen+Len];
                    currentPredictIds=forward(currentTagsScores,
                                             transitions,viterbi=True,
                                             return_alpha=False,
                                             return_best_sequence=True) ;
                    predictTag=T.set_subtensor(predictTag[accLen:accLen+Len],currentPredictIds);
                    return predictTag;
                    #}}}
                predictTag,update=theano.scan(fn=sentVitebe,
                                             outputs_info=T.zeros((tags_scores.shape[0],),dtype='int32'),
                                             sequences=[T.arange(docLen.shape[0])],
                                             non_sequences=[tags_scores,self.transitions,docLen]);
                predictTag=predictTag[-1];
            else:
                predictTag=forward(tags_scores, self.transitions, 
                                   viterbi=True,return_alpha=False, 
                                   return_best_sequence=True) 
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=predictTag,
                on_unused_input='ignore',
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
            #f_AttenVisual=theano.function(
            #    inputs=eval_inputs,
            #    outputs=[predictTag,self.energy],
            #    on_unused_input='ignore',
            #    givens=({is_train: np.cast['int32'](0)} if dropout else {})
            #    )
            #self.f_AttenVisual=f_AttenVisual;

        return f_train, f_eval;
#}}}

    def build(self,parameters):
#{{{
        """
        Build the network.
        """
        #some parameters 
        dropout=parameters['dropout'] ;
        char_dim=parameters['char_dim'];
        char_lstm_dim=parameters['char_lstm_dim'];
        char_bidirect=parameters['char_bidirect'];
        word_dim=parameters['word_dim'];
        word_lstm_dim=parameters['word_lstm_dim'];
        word_bidirect=parameters['word_bidirect'];
        lr_method=parameters['lr_method'];
        pre_emb=parameters['pre_emb'];
        crf=parameters['crf'];
        cap_dim=parameters['cap_dim'];
        training=parameters['training'];
        features=parameters['features'];
        
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)
        self.output_dim = len(self.id_to_tag);
        self.transitions = shared((self.output_dim+ 1, self.output_dim ), 'transitions')

        # Number of capitalization features
        if cap_dim:
            n_cap = 4
        
        if features is not None and features['lemma']['isUsed']:
            lemma_ids=T.ivector(name='lemma_ids');
        if features is not None and features['pos']['isUsed']:
            pos_ids=T.ivector(name='pos_ids');
        if features is not None and features['chunk']['isUsed']:
            chunk_ids=T.ivector(name='chunk_ids');
        if features is not None and features['NER']['isUsed']:
            dic_ids=T.ivector(name='dic_ids');

        # Network variables
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        tag_ids = T.ivector(name='tag_ids')
        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        # Word inputs
#{{{
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            #for attention 
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print 'Loading pretrained embeddings from %s...' % pre_emb
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print 'WARNING: %i invalid lines' % emb_invalid
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in xrange(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1
                word_layer.embeddings.set_value(new_weights)
                print 'Loaded %i pretrained embeddings.' % len(pretrained)
                print ('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros, n_words,
                            100. * (c_found + c_lower + c_zeros) / n_words
                      )
                print ('%i found directly, %i after lowercasing, '
                       '%i after lowercasing + zero.') % (
                          c_found, c_lower, c_zeros
                      )#}}}

        # Chars inputs
#{{{
        if char_dim:
            input_dim += char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            char_lstm_for.link(char_layer.link(char_for_ids))
            char_lstm_rev.link(char_layer.link(char_rev_ids))

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]

            inputs.append(char_for_output)
            if char_bidirect:
                inputs.append(char_rev_output)
                input_dim += char_lstm_dim
#}}}
        
        # Capitalization feature
        #
        if cap_dim:
            input_dim += cap_dim
            cap_layer = EmbeddingLayer(n_cap, cap_dim, name='cap_layer')
            inputs.append(cap_layer.link(cap_ids))

        # Prepare final input
        if len(inputs) != 1:
            inputs = T.concatenate(inputs, axis=1)

        #
        # Dropout on final input
        #
        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for')
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev')
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])
        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]
        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer', activation='tanh')
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        # Sentence to Named Entity tags - Score
        final_layer = HiddenLayer(word_lstm_dim, n_tags, name='final_layer',
                                  activation=(None if crf else 'softmax'))
        tags_scores = final_layer.link(final_output)

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()
        # CRF
        else:

            #all_paths_scores = forward(observations, self.transitions)
            #cost = - (self.modelScore(tag_ids,tags_scores,s_len) - all_paths_scores)
            #real_path_score=self.modelScore(tag_ids,tags_scores,tag_ids.shape[0]) ;
            #error=real_path_score+self.noiseLoss(tags_scores,tag_ids,0.5);
            #cost=-error;
            #cost=self.likehoodLoss(tags_scores,tag_ids,observations,2)
            
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

            # Score from transitions
            padded_tags_ids = T.concatenate([[n_tags], tag_ids], axis=0)
            real_path_score += self.transitions[
                padded_tags_ids[T.arange(s_len )],
                padded_tags_ids[T.arange(s_len ) + 1]
            ].sum()

            all_paths_scores = forward(tags_scores, self.transitions)
            cost = - (real_path_score - all_paths_scores)

        # Network parameters
        params = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            if char_bidirect:
                self.add_component(char_lstm_rev)
                params.extend(char_lstm_rev.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)
        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)
        if cap_dim:
            self.add_component(cap_layer)
            params.extend(cap_layer.params)
        self.add_component(final_layer)
        params.extend(final_layer.params)
        if crf:
            self.add_component(self.transitions)
            params.append(self.transitions)
        if word_bidirect:
            self.add_component(tanh_layer)
            params.extend(tanh_layer.params)

        # Prepare train and eval inputs
        eval_inputs = []
        if word_dim:
            eval_inputs.append(word_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        if cap_dim:
            eval_inputs.append(cap_ids)
        train_inputs = eval_inputs + [tag_ids]

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function
        print 'Compiling...'
        if training:
            import  optimizers ;
            self.optimizer=optimizers.RMSprop(lr=0.001);
            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)
            self.constraints={};
            #updates = self.optimizer.get_updates(params,self.constraints,cost);
            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )
            #for debug 
            #f_Debug = theano.function(
            #    inputs=train_inputs,
            #    outputs=cost,
            #    updates=self.update,
            #    givens=({is_train: np.cast['int32'](1)} if dropout else {})
            #)
            #debug end 
        else:
            f_train = None

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(tags_scores, self.transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )

        return f_train, f_eval
#}}}
