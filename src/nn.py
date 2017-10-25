import theano
import theano.tensor as T
from utils import shared
import numpy as np

class HiddenLayer(object): 
#{{{
    """
    Hidden layer with or without bias.
    Input: tensor of dimension (dims*, input_dim)
    Output: tensor of dimension (dims*, output_dim)
    """
    def __init__(self, input_dim, output_dim, bias=True, activation='sigmoid',
                 name='hidden_layer'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.name = name
        if activation is None:
            self.activation = None
        elif activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'softmax':
            self.activation = T.nnet.softmax
        else:
            raise Exception("Unknown activation function: " % activation)

        # Initialize weights and bias
        self.weights = shared((input_dim, output_dim), name + '_weights')
        self.bias = shared((output_dim,), name + '_bias')

        # Define parameters
        if self.bias:
            self.params = [self.weights, self.bias]
        else:
            self.params = [self.weights]

    def link(self, input):
        """
        The input has to be a tensor with the right
        most dimension equal to input_dim.
        """
        self.input = input
        self.linear_output = T.dot(self.input, self.weights)
        if self.bias:
            self.linear_output = self.linear_output + self.bias
        if self.activation is None:
            self.output = self.linear_output
        else:
            self.output = self.activation(self.linear_output)
        return self.output
#}}}

class EmbeddingLayer(object):
#{{{
    """
    Embedding layer: word embeddings representations
    Input: tensor of dimension (dim*) with values in range(0, input_dim)
    Output: tensor of dimension (dim*, output_dim)
    """

    def __init__(self, input_dim, output_dim, name='embedding_layer'):
        """
        Typically, input_dim is the vocabulary size,
        and output_dim the embedding dimension.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name

        # Randomly generate weights
        self.embeddings = shared((input_dim, output_dim),
                                 self.name + '__embeddings')

        # Define parameters
        self.params = [self.embeddings]

    def link(self, input):
        """
        Return the embeddings of the given indexes.
        Input: tensor of shape (dim*)
        Output: tensor of shape (dim*, output_dim)
        """
        self.input = input
        self.output = self.embeddings[self.input]
        return self.output
#}}}


class DropoutLayer(object):
#{{{
    """
    Dropout layer. Randomly set to 0 values of the input
    with probability p.
    """
    def __init__(self, p=0.5, name='dropout_layer'):
        """
        p has to be between 0 and 1 (1 excluded).
        p is the probability of dropping out a unit, so
        setting p to 0 is equivalent to have an identity layer.
        """
        assert 0. <= p < 1.
        self.p = p
        self.rng = T.shared_randomstreams.RandomStreams(seed=123456)
        self.name = name

    def link(self, input):
        """
        Dropout link: we just apply mask to the input.
        """
        if self.p > 0:
            mask = self.rng.binomial(n=1, p=1-self.p, size=input.shape,
                                     dtype=theano.config.floatX)
            self.output = input * mask
        else:
            self.output = input

        return self.output
#}}}

from keras import activations;
from keras import backend as K;
from keras import initializers as initializations;

class Layer(object):
    def __init__(self):
        self.build();
        return;
    def build(self):
        return;

class Convolution1D(Layer):
#{{{
    def __init__(self,nb_filter,filter_length,input_dim,init='glorot_uniform',
                    activation=None,border_mode='valid',subsample_length=1,
                    bias=True,
                    name='Convolution1D'):
#{{{
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.init = initializations.get(init, dim_ordering='th')
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample_length = subsample_length

        self.subsample = (subsample_length, 1) 
        self.bias=bias;
        self.input_dim = input_dim
        self.name=name;

        super(Convolution1D,self).__init__();
#}}}
    def build(self):
#{{{
        self.W_shape=(self.filter_length,1,self.input_dim,self.nb_filter);
        
        self.W=self.init(self.W_shape,name='{}_W'.format(self.name));
        if self.bias:
            init=initializations.get('zero');
            self.b=init((self.nb_filter,),
                                name='{}_b'.format(self.name));

        self.params=[self.W,self.b];
#}}}
    def call(self,x):
#{{{
        x=K.expand_dims(x,0);
        x=K.expand_dims(x,2);
        output=K.conv2d(x,self.W,strides=self.subsample,
                            border_mode=self.border_mode,
                            dim_ordering='tf');
        output=K.squeeze(output,2);
        if self.bias:
            output+=K.reshape(self.b,(1,1,self.nb_filter));
        output=self.activation(output);
        output=K.squeeze(output,0);
        return output;
#}}}
#}}}

class LSTM(object): 
#{{{
#{{{
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """
#}}}
    def __init__(self, input_dim, output_dim, with_batch=True, 
                 activation='tanh',inner_activation='hard_sigmoid',
                 name='LSTM'):
#{{{
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim;
        self.with_batch = with_batch
        self.name = name 
        self.inner_activation=activations.get(inner_activation);
        self.activation=activations.get(activation);
        self.build();
#}}}
    def build(self):
#{{{
        self.W=shared((self.input_dim,self.output_dim*3),name='{}_W'.format(self.name));
        self.U=shared((self.output_dim,self.output_dim*3),name='{}_U'.format(self.name));
        self.w_ci = shared((self.output_dim, self.output_dim), name='{}_w_ci'.format(self.name)  )
        self.w_co = shared((self.output_dim, self.output_dim), name='{}_w_co'.format(self.name)  )
        self.b=shared((self.output_dim*3,),name='{}_b'.format(self.name));
        self.c_0 = shared((self.output_dim,), name='{}_c_0'.format(self.name)  )
        self.h_0 = shared((self.output_dim,), name='{}_h_0'.format(self.name)  )
        self.params=[self.W,self.U,
                                    self.w_ci,self.w_co,self.b,
                                    self.c_0,self.h_0,
                    ];
        #}}}
   
    def get_initial_states(self, x):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.output_dim])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        return initial_states 
    def step(self,x, h_tm1,c_tm1):
#{{{
        z=T.dot(x,self.W)+T.dot(h_tm1,self.U)+self.b;
        if self.with_batch:
            z_i=z[:,:self.output_dim];
            z_c=z[:,self.output_dim:2*self.output_dim];
            z_o=z[:,2*self.output_dim:];
        else:
            z_i=z[:self.output_dim];
            z_c=z[self.output_dim:2*self.output_dim];
            z_o=z[2*self.output_dim:];

        i_t = self.inner_activation(z_i +
                                 T.dot(c_tm1, self.w_ci))
        # f_t = T.nnet.sigmoid(T.dot(x_t, self.w_xf) +
        #                      T.dot(h_tm1, self.w_hf) +
        #                      T.dot(c_tm1, self.w_cf) +
        #                      self.b_f)
        c_t = (1 - i_t) * c_tm1 + i_t * self.activation(z_c)
        o_t = self.inner_activation(z_o +
                                 T.dot(c_t, self.w_co))
        h_t = o_t * self.activation(c_t)
        return  h_t,c_t
#}}}
    def link(self, input):
#{{{
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """

        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            initial_states = [T.alloc(x, self.input.shape[1], self.output_dim)
                            for x in [self.h_0, self.c_0]]
        else:
            self.input = input
            initial_states = [self.h_0, self.c_0] 
        
        [h,c], _ = theano.scan(
            fn=self.step,
            sequences=self.input,
            outputs_info=initial_states,
        )
        self.h = h
        self.c=c
        self.output = h[-1]

        return self.output
#}}}
#}}}

class LSTM_normal(object): 
#{{{
#{{{
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """
#}}}
    def __init__(self, input_dim, output_dim, with_batch=True, 
                 activation='tanh',inner_activation='hard_sigmoid',
                 name='LSTM_normal'):
#{{{
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim;
        self.with_batch = with_batch
        self.name = name 
        self.inner_activation=activations.get(inner_activation);
        self.forget_bias_init = initializations.get('one')
        self.activation=activations.get(activation);
        self.build();
#}}}
    def build(self):
#{{{
        import numpy as np;
        self.W = shared((self.input_dim, 4 * self.output_dim),
                               name='{}_W'.format(self.name))
        self.U = shared((self.output_dim, 4 * self.output_dim),
                                     name='{}_U'.format(self.name))

        self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                        K.get_value(self.forget_bias_init(
                                                (self.output_dim,))),
                                        np.zeros(self.output_dim),
                                        np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))
        #self.c_0 = shared((self.output_dim,), name='{}_c_0'.format(self.name)  )
        #self.h_0 = shared((self.output_dim,), name='{}_h_0'.format(self.name)  )
        self.c_0=np.zeros(self.output_dim).astype(theano.config.floatX);
        self.h_0=np.zeros(self.output_dim).astype(theano.config.floatX);
        self.params=[self.W,self.U,
                        self.b,
                    # self.c_0,self.h_0
                    ];
        #}}}
    def step(self,x, h_tm1,c_tm1):
#{{{
        z = K.dot(x , self.W) + K.dot(h_tm1 , self.U) + self.b
        if self.with_batch:
            z0 = z[:,:self.output_dim]
            z1 = z[:,self.output_dim: 2 * self.output_dim]
            z2 = z[:,2 * self.output_dim: 3 * self.output_dim]
            z3 = z[:,3 * self.output_dim:]
        else:
            z0 = z[:self.output_dim]
            z1 = z[self.output_dim: 2 * self.output_dim]
            z2 = z[2 * self.output_dim: 3 * self.output_dim]
            z3 = z[3 * self.output_dim:]


        i = self.inner_activation(z0)
        f = self.inner_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.inner_activation(z3) 
        h=o*self.activation(c);
        return  h,c;
#}}}
    
    def link(self, input):
#{{{
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """

        # If we use batches, we have to permute the first and second dimension.
        self.input = input
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            initial_states = [T.alloc(x, self.input.shape[1], self.output_dim)
                            for x in [self.h_0, self.c_0]]
        else:
            self.input = input
            initial_states = [self.h_0, self.c_0] 
        step_function=self.step;

        [h,c], _ = theano.scan(
            fn=step_function,
            sequences=self.input,
            outputs_info=initial_states,
        )
        self.h = h
        self.output = h[-1]

        return self.output
#}}}
#}}}

class AttentionLSTM(LSTM):
    def build(self):
#{{{
        super(AttentionLSTM,self).build()   ;
        self.W_A=shared((self.input_dim+self.output_dim,1),name='{}_W_A'.format(self.name));
        self.b_A=shared((1,),name='{}_b_A'.format(self.name));
        self.params+=[self.W_A,self.b_A];
#}}}
    def step(self, h_tm1,c_tm1,x):
#{{{
        assert x.ndim==2;
        H=x;
        input_length=x.shape[0];
        C=T.repeat(c_tm1.reshape((1,-1)),input_length,axis=0);
        _HC=K.concatenate([H,C]);
        energy=T.dot(_HC,self.W_A.reshape((-1,1)))+self.b_A;
        energy=K.softmax(energy.reshape((1,-1)));
        x=(H*energy.reshape((-1,1))).sum(axis=0)
        
        h_t,c_t=super(AttentionLSTM,self).step_noBatch(x,h_tm1,c_tm1);
        return  h_t,c_t
#}}}
    def link(self, input):
#{{{
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """

        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            assert 0,"AttentionLSTM not implement with_batch";
        else:
            self.input=input;
            initial_states = [self.h_0, self.c_0] 
         
        step_function=self.step;  

        [h,c], _ = theano.scan(
            fn=step_function,
            outputs_info=initial_states,
            non_sequences=[self.input],
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = self.h[-1]

        return self.output
#}}}
 
class AttentionLSTM2(AttentionLSTM):
#{{{
    def __init__(self,attended_dim,wordInput_dim,
                 combineOput_dim,output_dim, **kwargs):
#{{{
        self.attendedInput_dim=attended_dim;
        self.wordInput_dim=wordInput_dim;
        self.combineOput_dim=combineOput_dim;
        super(AttentionLSTM2, self).__init__(output_dim=output_dim,
                                             input_dim=combineOput_dim,
                                             **kwargs)
#}}}
    def build(self):
#{{{
        if self.input_dim is None:
            self.input_dim=self.combineOput_dim;
        super(AttentionLSTM,self).build()   ;
        #attention weight
        self.W_A=shared((self.attendedInput_dim+self.output_dim,1),name='{}_W_A'.format(self.name));
        self.b_A=shared((1,),name='{}_b_A'.format(self.name));
        
        #combine weight
        self.W_combine=shared((self.attendedInput_dim+self.wordInput_dim,
                                 self.combineOput_dim),
                                 name='{}_W_combine'.format(self.name));
        self.b_combine=shared((self.combineOput_dim,),
                                 name='{}_b_combine'.format(self.name));
        self.params+=[self.W_A,self.b_A];
        self.params+=[self.W_combine,self.b_combine];

#}}}
    def step(self, word,h_tm1,c_tm1,x):
#{{{
        H=x;
        input_length=x.shape[0];
        C=T.repeat(c_tm1.reshape((1,-1)),input_length,axis=0);
        _HC=K.concatenate([H,C]);
        energy=T.dot(_HC,self.W_A.reshape((-1,1)))+self.b_A;
        energy=K.softmax(energy.reshape((1,-1)));
        x=(H*energy.reshape((-1,1))).sum(axis=0)

        #combine glimpsed with word;
        combine=K.concatenate([x,word]);
        combined=K.dot(combine,self.W_combine)+self.b_combine;
        #original LSTM step
        h_t,c_t=super(AttentionLSTM,self).step_noBatch(combined,h_tm1,c_tm1);
        return  h_t,c_t
#}}}
    def link(self, input,words):
#{{{
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """

        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            assert 0,"AttentionLSTM not implement with_batch";
        else:
            self.input = input
            initial_states = [self.h_0, self.c_0] 
        
        step_function=self.step;  

        [h,c], _ = theano.scan(
            fn=step_function,
            sequences=[words],
            outputs_info=initial_states,
            non_sequences=[self.input],
        )
        self.h = h
        self.output = h[-1]

        return self.output
#}}}
 
#}}}

class AttentionLSTM3(LSTM):
#{{{
    def __init__(self,attended_dim,wordInput_dim,
                 output_dim,mode='concat', **kwargs):
#{{{
        self.attendedInput_dim=attended_dim;
        self.wordInput_dim=wordInput_dim;
        self.attendedMode=mode;
        self.init=initializations.get('glorot_uniform');
        super(AttentionLSTM3, self).__init__(output_dim=output_dim,
                                             input_dim=attended_dim+wordInput_dim,
                                             **kwargs)
#}}}
    def build(self):
#{{{
        if self.input_dim is None:
            self.input_dim=self.combineOput_dim;
        super(AttentionLSTM3,self).build()   ;
        #attention weight 
        self.W_A_X=shared((self.attendedInput_dim,self.output_dim),
                             name='{}_W_A_X');
        #self.b_A_X=shared((self.output_dim,),
        #                     name='{}_b_A_X');
        self.W_A_h=shared((self.output_dim,self.output_dim),
                             name='{}_W_A_h');
        #self.b_A_h=shared((self.output_dim,),
        #                     name='{}_b_A_h');
        self.W_A=self.init((self.output_dim,),name='{}_W_A'.format(self.name));
        #self.b_A=shared((1,),name='{}_b_A'.format(self.name));
        self.params+=[self.W_A_X,
                      #self.b_A_X,
                          self.W_A_h,
                      #self.b_A_h,
                            self.W_A,
                      #self.b_A,
                         ];


#}}}
    def step(self, word,index,energy_tm1,h_tm1,c_tm1,x):
#{{{
        #attention 
        H=x;
        if self.attendedMode is "concat":
            M_X=T.dot(x,self.W_A_X)#+self.b_A_X;
            M_state=T.dot(self.W_A_h,c_tm1)#+self.b_A_h; 
            M=T.tanh(M_X+M_state)
            _energy=T.dot(M,self.W_A.T)#+self.b_A;
        elif self.attendedMode is "dot":
            energy=None;
            assert 0,"not implement";
        elif self.attendedMode is "general":
            M_X=T.dot(x,self.W_A_X)#+self.b_A_X;
            M_state=T.dot(self.W_A_h,c_tm1)#+self.b_A_h; 
            M=T.tanh(M_X*M_state);
            _energy=T.dot(M,self.W_A.T)#+self.b_A;
        #mask
        mask=T.zeros((1,x.shape[0]),dtype=theano.config.floatX);
        energy=T.nnet.softmax(_energy[:index+1]);
        masked_energy=T.set_subtensor(mask[0,:index+1],energy.flatten());
        glimpsed=(masked_energy.T*H).sum(axis=0)
        #combine glimpsed with word;
        if self.wordInput_dim==0:
            combined=glimpsed;
        else:
            combine=K.concatenate([glimpsed,word]);
            combined=combine; 
        #original LSTM step 
        h_t,c_t=super(AttentionLSTM3,self).step(combined,h_tm1,c_tm1);
        return  masked_energy.flatten(),h_t,c_t
#}}}
    def link(self, input,words):
#{{{
        """
        Propagate the input through the network and return the last hidden
        vector. The whole sequence is also accessible via self.h, but
        where self.h of shape (sequence_length, batch_size, output_dim)
        """

        # If we use batches, we have to permute the first and second dimension.
        if self.with_batch:
            assert 0,"AttentionLSTM not implement with_batch";
        else:
            self.input = input
            initial_states = [self.h_0, self.c_0] 
        
        step_function=self.step;  

        [e,h,c], _ = theano.scan(
            fn=step_function,
            sequences=[words,T.arange(words.shape[0])],
            outputs_info=[T.zeros((input.shape[0],),
                                  dtype=theano.config.floatX)]+initial_states,
            non_sequences=[self.input],
        )
        self.h = h
        self.output = h[-1]
        self.e=e;
        self.c=c;
        return self.output
#}}}
 
#}}}

class AttentionLayer(Layer):
    def __init__(self,attended_dim,state_dim,
                source_dim,scoreFunName='Euclidean',
                 atten_activation='tanh',name='AttentionLayer'):
#{{{
        self.attended_dim=attended_dim;
        self.state_dim=state_dim;
        self.source_dim=source_dim;
        self.init=initializations.get('glorot_uniform');
        self.name=name;
        self.one_init=initializations.get('one');
        self.atten_activation=activations.get(atten_activation);
        self.scoreFunName=scoreFunName;
        self.eps=1e-5;
        #self.source_dim=glimpsed_dim;
        super(AttentionLayer,self).__init__();
    #}}}
    def euclideanScore(self,attended,state,W):
#{{{
        #Euclidean distance 
        M=(attended-state)**2;
        M=T.dot(M,W);
        _energy=M.max()-M;
        return _energy; 
#}}}
    def manhattenScore(self,attended,state,W):
#{{{
        #Manhattan Distance 
        #eps for avoid gradient to be NaN;
        M=T.abs_(T.maximum(attended-state,self.eps));
        M=T.dot(M,W);
        _energy=M.max()-M;
        return _energy; 
#}}}
    def bilinearScore(self,attended,state,W):
#{{{
        #Bilinear function  
        M=(attended*state*W).sum(axis=-1);
        _energy=self.atten_activation(M);
        return _energy;
#}}}
    def forwardNNScore(self,attended,state,W):
#{{{
        #get weights
        W_1=W[:(self.attended_dim+self.state_dim)*self.state_dim]; 
        W_1=W_1.reshape((self.attended_dim+self.state_dim,self.state_dim));
        W_2=W[(self.attended_dim+self.state_dim)*self.state_dim:];
        
        #forward neural network 
        state_=T.repeat(state.reshape((1,-1)),attended.shape[0],axis=0);
        input=T.concatenate([attended,state_],axis=-1);
        M1=self.atten_activation(T.dot(input,W_1));
        M2=self.atten_activation(T.dot(M1,W_2));
        _energy=M2;
        return _energy;
    #}}}
    def CNNScore(self,attended,state,W):
#{{{
        state_=T.repeat(state.reshape((1,-1)),attended.shape[0],axis=0);
        input=T.concatenate([attended,state_],axis=-1);
        M1=self.CNN1.call(input);
        M2=self.CNN2.call(M1);
        _energy=M2.flatten();
        return _energy;
#}}}
    def CosineScore(self,attended,state,W):
#{{{
        dotProduct=T.dot(attended,state.T);
        Al2Norm=T.sqrt((attended**2).sum(axis=-1));
        Bl2Norm=T.sqrt((state**2).sum(axis=-1));
        M=dotProduct/(Al2Norm*Bl2Norm);
        _energy=T.exp(M+2);
        return _energy;
#}}}
    def vanilaScore(self,attended,state,W):
        """
            the origin score proprosed by Bahdanau 2015
        """

    def build(self):
#{{{
        self.W_A_X=shared((self.attended_dim,self.attended_dim),
                             name='{}_W_A_X'.format(self.name));
        self.b_A_X=shared((self.attended_dim,),
                            name='{}_W_A_b'.format(self.name));
        self.W_A_h=shared((self.attended_dim,self.attended_dim),
                             name='{}_W_A_h'.format(self.name));
        self.W_A_combine=shared((self.source_dim*2,
                                 self.source_dim),
                               name='{}_W_A_combine'.format(self.name));
        self.b_A_combine=shared((self.source_dim,),
                               name='{}_b_A_combine'.format(self.name))
        #self.W_A_combine=shared((self.source_dim,
        #                         self.source_dim),
        #                         name='{}_W_A_combine'.format(self.name));
        #self.b_A_combine=shared((self.source_dim,),
        #                         name='{}_b_A_combine'.format(self.name))
        #use constraint
        self.constraints={}
        
        self.params=[
                     self.W_A_X,self.b_A_X,
                    # self.W_A_h,
                     self.W_A_combine,self.b_A_combine
                    ];
        
        #for attention weight and score function
        if self.scoreFunName == "Euclidean":
#{{{
            self.W_A=shared((self.state_dim,),
                          name='{}_W_A'.format(self.name));
            self.W_A.set_value(np.ones((self.state_dim,),dtype=theano.config.floatX));
            self.constraints[self.W_A]=self.NonNegConstraint;
            self.scoreFun=self.euclideanScore;
            self.params.append(self.W_A);
#}}}
        elif self.scoreFunName == "Bilinear":
#{{{
            assert self.attended_dim==self.state_dim,"in Bilinear score function,"\
                " attended_dim must be equal to state_dim"
            self.W_A=self.init((self.state_dim,),
                                name="{}_W_A".format(self.name));
            self.scoreFun=self.bilinearScore;
            self.params.append(self.W_A);
#}}}
        elif self.scoreFunName == "forwardNN":
#{{{
            #this is two layer NN 
            #first layer (attended_dim+state_dim,state_dim);
            #second layer (state_dim,1);
            self.W_A=shared(((self.attended_dim+self.state_dim)\
                                *self.state_dim+self.state_dim,),
                                name="{}_W_A".format(self.name));
            self.scoreFun=self.forwardNNScore;
            self.params.append(self.W_A);
#}}}
        elif self.scoreFunName == "CNN":
#{{{
            #this if one layer CNN and pool layer;
            nb_filter=(self.attended_dim+self.state_dim)/2;
            filter_length=3;
            input_dim=self.attended_dim+self.state_dim;
            self.CNN1=Convolution1D(nb_filter=nb_filter,
                                   filter_length=filter_length,
                                  input_dim=input_dim,activation='tanh',
                                  border_mode='same');
            self.CNN2=Convolution1D(nb_filter=1,
                                   filter_length=filter_length,
                                  input_dim=nb_filter,activation='tanh',
                                  border_mode='same');
            self.W_A=self.CNN1.W;
            self.scoreFun=self.CNNScore;
            self.params.append(self.W_A);
            self.params.append(self.CNN2.W);
#}}}
        elif self.scoreFunName == "Cosine":
#{{{
            self.scoreFun=self.CosineScore;
            self.W_A=None;
#}}}
        elif self.scoreFunName == "Manhatten":
#{{{
            self.scoreFun=self.manhattenScore;
            self.W_A=self.one_init((self.state_dim,),
                          name='{}_W_A'.format(self.name));
            self.constraints[self.W_A]=self.NonNegConstraint;
            self.params.append(self.W_A);
#}}}
        else:
            assert 0, "we only have Euclidean, Bilinear, forwardNN"\
                    " score function for attention";

#}}}
    def softmaxReScale(self,energy_,threshould):
#{{{
        #in energy_, the goundthrud should be max
        assert energy_.ndim==1;
        #convert threshould from percentage to energy_;
        threshould_=T.log(T.exp(energy_-energy_.max()).sum())+T.log(threshould)+energy_.max()
        energy=self.reScale(energy_,threshould_);
        return T.nnet.softmax(energy);
    #}}}
    def reScale(self,energy,threshold,replaceValue=1e-7):
#{{{
        assert energy.ndim==1;
        maxValue=energy.max();
        def checkThreshold(value,threshold,replaceValue):
            return T.switch(T.lt(value,threshold),replaceValue,value);
        result,update=theano.scan(fn=checkThreshold,
                                 outputs_info=None,
                                 sequences=[energy],
                                 non_sequences=[threshold,replaceValue]);
        return T.switch(T.lt(maxValue,threshold),energy,result);
#}}}
    
    def step(self,state,attended,source):
        #from theano.gradient import disconnected_grad;
        #state=disconnected_grad(state_);
        #M_state=T.dot(self.W_A_h,state) ;

        _energy=self.scoreFun(attended,state,self.W_A)
        energy=T.nnet.softmax(_energy);
        #energy=self.softmaxReScale(_energy,0.02);
        #energy=self.reScale(energy.flatten(),0.02).reshape((1,-1))
        #energyIndex=energy.flatten().argmin(axis=-1);
        glimpsed=(energy.T*source).sum(axis=0)
        #glimpsed=source[energyIndex];
        return energy.flatten(),glimpsed;

    def NonNegConstraint(self,p):
        p*=K.cast(p>=0.,K.floatx());
        return p;

    def link(self,attended,state,source):
        step_function=self.step;
        attended_=T.tanh(T.dot(attended,self.W_A_X))+self.b_A_X;
        #attended_=attended;
        [energy,glimpsed],_=theano.scan(fn=step_function,
                            sequences=[attended_],
                               outputs_info=None,
                            non_sequences=[attended_,source]);
        self.energy=energy;
        
        #combine 
        #combine=T.concatenate([glimpsed,attended],axis=-1);
        combine=T.concatenate([glimpsed,source],axis=-1);
        combined=T.tanh(T.dot(combine,self.W_A_combine))+self.b_A_combine;
        #no source
        #combined=T.tanh(T.dot(glimpsed,self.W_A_combine))+self.b_A_combine;
        return combined;

def log_sum_exp(x, axis=None):
    """
    Sum probabilities in the log-space.
    """
    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

def forward(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)
    """
    #the last row of transitions is the inital state 
    trans=transitions[:-1];
    assert not return_best_sequence or (viterbi and not return_alpha)
    assert viterbi==return_best_sequence

    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            
            out2 = scores.argmax(axis=0)
            return out, out2
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = transitions[-1]+observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=[trans]
    )
    if viterbi:
            alpha0=T.concatenate([[initial],alpha[0]],axis=0);
            alpha=[alpha0,alpha[1]];
    #else:
    #    alpha=T.concatenate([log_sum_exp(initial,axis=0).dimshuffle('x',0),
    #                            alpha],axis=0);

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1]), 'int32'),
            sequences=T.cast(alpha[1][::-1], 'int32')
        )
        sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1])]])
        return sequence
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)


def forward_org(observations, transitions, viterbi=False,
            return_alpha=False, return_best_sequence=False):
    """
    Takes as input:
        - observations, sequence of shape (n_steps, n_classes)
        - transitions, sequence of shape (n_classes, n_classes)
    Probabilities must be given in the log space.
    Compute alpha, matrix of size (n_steps, n_classes), such that
    alpha[i, j] represents one of these 2 values:
        - the probability that the real path at node i ends in j
        - the maximum probability of a path finishing in j at node i (Viterbi)
    Returns one of these 2 values:
        - alpha
        - the final probability, which can be:
            - the sum of the probabilities of all paths
            - the probability of the best path (Viterbi)

    """
    
    assert not return_best_sequence or (viterbi and not return_alpha)

    def recurrence(obs, previous, transitions):
        previous = previous.dimshuffle(0, 'x')
        obs = obs.dimshuffle('x', 0)
        if viterbi:
            scores = previous + obs + transitions
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)
                return out, out2
            else:
                return out
        else:
            return log_sum_exp(previous + obs + transitions, axis=0)

    initial = observations[0]
    alpha, _ = theano.scan(
        fn=recurrence,
        outputs_info=(initial, None) if return_best_sequence else initial,
        sequences=[observations[1:]],
        non_sequences=transitions
    )

    if return_alpha:
        return alpha
    elif return_best_sequence:
        sequence, _ = theano.scan(
            fn=lambda beta_i, previous: beta_i[previous],
            outputs_info=T.cast(T.argmax(alpha[0][-1]), 'int32'),
            sequences=T.cast(alpha[1][::-1], 'int32')
        )
        #sequence = T.concatenate([sequence[::-1], [T.argmax(alpha[0][-1])]])
        return alpha
    else:
        if viterbi:
            return alpha[-1].max(axis=0)
        else:
            return log_sum_exp(alpha[-1], axis=0)

