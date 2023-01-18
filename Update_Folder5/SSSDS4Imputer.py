import math
#----------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow_probability.python.internal import tensor_util
#from tensorflow import keras
#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
#import tensorflow_probability as tfp
#import tensorflow_addons as tfa
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#-----------------------------------------------------------------------------------
from utils.util import calc_diffusion_step_embedding
from imputers.S4Model import S4Layer


def swish(x):
    return x * tf.math.sigmoid(x)


class Conv(Layer):  # 这里我们继承于tf.keras.layers.Layer，和Pytorch的nn.Model()是等价替代关系
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, activation_fn="relu"):
        super(Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        #self.padding = dilation * (kernel_size - 1) // 2
        #self.units = units

    def build(self, kernel_size=3, dilation=1, activation_fn="relu"):
        #self.w = self.add_weight(name ="w",shape=(input_shape[-1],self.units), initializer=tf.keras.initializers.HeNormal(),trainable=True)
        #self.b = self.add_weight(name ="b",shape=(self.units,), initializer=tf.keras.initializers.HeNormal(),trainable=True)
        self.conv = Conv1D(data_format='channels_first', filters=self.out_channels, kernel_size=self.kernel_size, dilation_rate=self.dilation, padding="same", activation=activation_fn, kernel_initializer=tf.keras.initializers.HeNormal(), use_bias=True, bias_initializer=tf.keras.initializers.HeNormal()) # padding="same" # input_shape=(in_channels,),

        #self.conv = tf.keras.Sequential()
        ##self.conv.add(tf.keras.Input(shape=(in_channels,)))
        #self.conv.add(Conv1D(data_format='channels_first', filters=out_channels, kernel_size=kernel_size, dilation_rate=dilation, padding="same", activation="relu", kernel_initializer=tf.keras.initializers.HeNormal())) # padding="same" # input_shape=(in_channels,),
        #self.conv = tfp.layers.weight_norm.WeightNorm(self.conv)  # 上面的groups是在tf.keras.layers.Conv1D中表示输入的input channels的数量
        

    def call(self, inputs):    # TF中的call就等于torch中的forward
        #--------------------------------------
        inputs = tensor_util.convert_nonref_to_tensor(inputs,dtype=tf.float32)
        #--------------------------------------
        out = self.conv(inputs)
        #--------------------------------------
        out = tensor_util.convert_nonref_to_tensor(out,dtype=tf.float32)
        #--------------------------------------
        return out
    
    
class ZeroConv1d(Layer):
    def __init__(self, in_channel, out_channel, activation_fn=None):
        super(ZeroConv1d, self).__init__()
        self.conv = Conv1D(data_format='channels_first', filters=out_channel, kernel_size=1, padding="same", activation=activation_fn, kernel_initializer=tf.keras.initializers.HeNormal(), use_bias=True, bias_initializer=tf.keras.initializers.HeNormal()) # kernel_initializer=tf.keras.initializers.Zeros() # bias_initializer=tf.keras.initializers.Zeros()
        #self.conv = tf.keras.Sequential()
        ##self.conv.add(tf.keras.Input(shape=(in_channel,)))
        #self.conv.add(Conv1D(data_format='channels_first', filters=out_channel, kernel_size=1, padding="same")) # input_shape=(in_channel,),
        #initializer = tf.keras.initializers.Zeros()
        #self.conv.weight = initializer
        #self.conv.bias = initializer

    def call(self, x):
        #--------------------------------------
        x = tensor_util.convert_nonref_to_tensor(x,dtype=tf.float32)
        #--------------------------------------
        out = self.conv(x)
        #--------------------------------------
        out = tensor_util.convert_nonref_to_tensor(out,dtype=tf.float32)
        #--------------------------------------
        return out


class Residual_block(Layer):
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
                s4_lmax,
                s4_d_state,
                s4_dropout,
                s4_bidirectional,
                s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels


        self.fc_t = tf.keras.Sequential()
        self.fc_t.add(tf.keras.Input(shape=(diffusion_step_embed_dim_out,)))
        self.fc_t.add(Dense(self.res_channels, activation=None, use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal())) # input_shape=(diffusion_step_embed_dim_out,),
        
        self.S41 = S4Layer(features=2*self.res_channels,                             #********
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
 
        self.conv_layer = Conv(in_channels = self.res_channels, out_channels = 2 * self.res_channels, kernel_size=3, activation_fn=None)

        self.S42 = S4Layer(features=2*self.res_channels,                             #********
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        
        self.cond_conv = Conv(in_channels = self.res_channels, out_channels = 2*self.res_channels, kernel_size=1, activation_fn=None)               

        self.res_conv = Conv1D(data_format='channels_first', filters=res_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(), use_bias=True, bias_initializer=tf.keras.initializers.HeNormal(), activation=None) # input_shape=(res_channels,),
        #self.res_conv = tf.keras.Sequential()
        ##self.res_conv.add(tf.keras.Input(shape=(res_channels,)))
        #self.res_conv.add(Conv1D(data_format='channels_first', filters=res_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal())) # input_shape=(res_channels,),
        #self.res_conv = tfp.layers.weight_norm.WeightNorm(self.res_conv)

        self.skip_conv = Conv1D(data_format='channels_first', filters=skip_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal(), use_bias=True, bias_initializer=tf.keras.initializers.HeNormal(), activation=None) # input_shape=(res_channels,),

        #self.skip_conv = tf.keras.Sequential()
        ##self.skip_conv.add(tf.keras.Input(shape=(res_channels,)))
        #self.skip_conv.add(Conv1D(data_format='channels_first', filters=skip_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal())) # input_shape=(res_channels,),
        #self.skip_conv = tfp.layers.weight_norm.WeightNorm(self.skip_conv)

    def call(self, input_data):                #----------------
        x, cond, diffusion_step_embed = input_data
        #----------------------------------------------
        x = tensor_util.convert_nonref_to_tensor(x,dtype=tf.float32)
        cond = tensor_util.convert_nonref_to_tensor(cond,dtype=tf.float32)
        diffusion_step_embed = tensor_util.convert_nonref_to_tensor(diffusion_step_embed,dtype=tf.float32)
        #----------------------------------------------
        h = x

        print("This is the type of h (1): --------------------------------",type(h))  #Add
        
        B, C, L = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t,[B, self.res_channels, 1])  
        h = h + part_t
        
        h = self.conv_layer(h)

        print("This is the type of h (2): --------------------------------",type(h))  #Add

        h = h.numpy()  # S4 imputer里面只能带入torch tensor所以我们这里将input的h调整为torch tensor带入下面S41
        h = torch.from_numpy(h)
        
        h = self.S41(h.permute(2,0,1)).permute(1,2,0)  #h = tf.transpose(self.S41(tf.transpose(h,perm=[2,0,1])),perm=[1,2,0])                               #********
        

        assert cond is not None
        cond = self.cond_conv(cond)

        cond = cond.numpy()   # 为了下面的torch tensor计算，这里把cond转为torch tensor
        cond = torch.from_numpy(cond)

        h += cond
        
        h = self.S42(h.permute(2,0,1)).permute(1,2,0)  #h = tf.transpose(self.S42(tf.transpose(h,perm=[2,0,1])),perm=[1,2,0])                               #********
        
        h = h.detach().numpy()
        h = tensor_util.convert_nonref_to_tensor(h,dtype=tf.float32)  # 在S42 imputer计算完毕后输出torch tensor,我们转为numpy 后继续转为 TF tensor

        out = tf.math.tanh(h[:,:self.res_channels,:]) * tf.math.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(Layer):                                       
    def __init__(self, res_channels, skip_channels, num_res_layers, 
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.diffusion_step_embed_dim_in = diffusion_step_embed_dim_in

        self.fc_t1 = layers.Dense(units=diffusion_step_embed_dim_mid, activation=None, use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal()) # input_shape=(diffusion_step_embed_dim_in,),

        self.fc_t2 = layers.Dense(units=diffusion_step_embed_dim_out, activation=None, use_bias=True, kernel_initializer=tf.keras.initializers.HeNormal(), bias_initializer=tf.keras.initializers.HeNormal()) # input_shape=(diffusion_step_embed_dim_mid,),
        
        # self.residual_blocks = []
        # for n in range(self.num_res_layers):
        self.residual_blocks = Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm)

            
    def call(self, input_data):
        noise, conditional, diffusion_steps = input_data
        #-----------------------------------------------
        noise = tensor_util.convert_nonref_to_tensor(noise,dtype=tf.float32)
        conditional = tensor_util.convert_nonref_to_tensor(conditional,dtype=tf.float32)
        diffusion_steps = tensor_util.convert_nonref_to_tensor(diffusion_steps,dtype=tf.int32)
        #------------------------------------------------

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = tensor_util.convert_nonref_to_tensor(diffusion_step_embed,dtype=tf.float32)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks((h, conditional, diffusion_step_embed))  
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)  


class SSSDS4Imputer(tf.keras.Model):                                        #===========================
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, 
                 num_res_layers,
                 diffusion_step_embed_dim_in, 
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(SSSDS4Imputer, self).__init__()

        #self.init_conv = tf.keras.Sequential()
        #self.init_conv.add(tf.keras.layers.Flatten())
        self.init_conv = Conv(in_channels = in_channels, out_channels = res_channels, kernel_size=1)  # 激活函数relu已经嵌入了上面的Conv() class，所以这里不用写
        
        
        self.residual_layer = Residual_group(res_channels=res_channels, 
                                             skip_channels=skip_channels, 
                                             num_res_layers=num_res_layers, 
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)
        
        self.final_conv = tf.keras.Sequential()
        #self.final_conv.add(tf.keras.Input(shape=(skip_channels,)))
        self.final_conv.add(Conv(in_channels = skip_channels, out_channels = skip_channels, kernel_size=1))
        self.final_conv.add(ZeroConv1d(in_channel = skip_channels, out_channel = out_channels))
        

    def call(self, input_data):
        
        noise, conditional, mask, diffusion_steps = input_data 

        conditional = conditional * mask
        conditional = tf.concat([conditional, mask], axis=1)
        #-------------------------------------------------------------
        noise = tensor_util.convert_nonref_to_tensor(noise,dtype=tf.float32)
        conditional = tensor_util.convert_nonref_to_tensor(conditional,dtype=tf.float32)
        mask = tensor_util.convert_nonref_to_tensor(mask,dtype=tf.float32)
        diffusion_steps = tensor_util.convert_nonref_to_tensor(diffusion_steps,dtype=tf.int32)
        #loss_mask = tensor_util.convert_nonref_to_tensor(loss_mask,dtype=tf.int32)
        #-------------------------------------------------------------
        x = noise
        #print('==============111',x) # Add
        #print('==============',input_data[2].shape) # Add

        x = self.init_conv(x)
        #print('==============222',x) # Add
        x = self.residual_layer((x, conditional, diffusion_steps))
        #print('==============333',x) # Add
        y = self.final_conv(x)
        #print('==============444',y) # Add

        #print('--------------',only_generate_missing)  #Add

        #if only_generate_missing[0] == 1:
        #  loss_mask = torch.tensor(loss_mask.numpy(),dtype=torch.bool)
        #  y = torch.tensor(y.numpy())
        #  y = tensor_util.convert_nonref_to_tensor(y[loss_mask].numpy().reshape((50,1260)),dtype=tf.float32) # tf.convert_to_tensor
        #  print('This is y: ++++++++++++++++++',y)
        #elif only_generate_missing[0] == 0:
        #  y = tensor_util.convert_nonref_to_tensor(y.numpy(),dtype=tf.float32)

        return y
