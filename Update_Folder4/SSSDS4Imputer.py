import math
#----------------------------------------------------------------------------------
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import torch
#import torch.nn as nn
#import torch.nn.functional as F
#-----------------------------------------------------------------------------------
from utils.util import calc_diffusion_step_embedding
from imputers.S4Model import S4Layer


def swish(x):
    return x * tf.math.sigmoid(x)


class Conv(tf.keras.layers.Layer):  # 这里我们继承于tf.keras.layers.Layer，和Pytorch的nn.Model()是等价替代关系
    def __init__(self, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = tf.keras.layers.Conv1D(data_format='channels_first', filters=out_channels, kernel_size=kernel_size, dilation_rate=dilation, padding="same", activation="relu", kernel_initializer=tf.keras.initializers.HeNormal()) # padding="same"
        #self.conv = tfp.layers.weight_norm.WeightNorm(self.conv)  # 上面的groups是在tf.keras.layers.Conv1D中表示输入的input channels的数量
        

    def call(self, inputs):    # TF中的call就等于torch中的forward
        out = self.conv(inputs)
        return out
    
    
class ZeroConv1d(tf.keras.layers.Layer):
    def __init__(self, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = tf.keras.layers.Conv1D(data_format='channels_first', filters=out_channel, kernel_size=1, padding="same")
        initializer = tf.keras.initializers.Zeros()
        self.conv.weight = initializer
        self.conv.bias = initializer

    def call(self, x):
        out = self.conv(x)
        return out


class Residual_block(tf.keras.layers.Layer):
    def __init__(self, res_channels, skip_channels, 
                 diffusion_step_embed_dim_out, in_channels,
                s4_lmax,
                s4_d_state,
                s4_dropout,
                s4_bidirectional,
                s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels


        self.fc_t = tf.keras.models.Sequential()
        self.fc_t.add(tf.keras.layers.Dense(self.res_channels, input_shape=(diffusion_step_embed_dim_out,), activation=None)) 
        
        self.S41 = S4Layer(features=2*self.res_channels,                             #********
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
 
        self.conv_layer = Conv(out_channels = 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels,                             #********
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        
        self.cond_conv = Conv(out_channels = 2*self.res_channels, kernel_size=1)               

        self.res_conv = tf.keras.layers.Conv1D(data_format='channels_first', filters=res_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal())
        #self.res_conv = tfp.layers.weight_norm.WeightNorm(self.res_conv)

        
        self.skip_conv = tf.keras.layers.Conv1D(data_format='channels_first', filters=skip_channels, kernel_size=1, kernel_initializer=tf.keras.initializers.HeNormal())
        #self.skip_conv = tfp.layers.weight_norm.WeightNorm(self.skip_conv)

    def call(self, input_data):                #----------------
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        part_t = tf.reshape(part_t,[B, self.res_channels, 1])  
        h = h + part_t
        
        h = self.conv_layer(h)

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
        h = tf.convert_to_tensor(h)  # 在S42 imputer计算完毕后输出torch tensor,我们转为numpy 后继续转为 TF tensor

        out = tf.math.tanh(h[:,:self.res_channels,:]) * tf.math.sigmoid(h[:,self.res_channels:,:])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability


class Residual_group(tf.keras.layers.Layer):                                       
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

        self.fc_t1 = tf.keras.models.Sequential()
        self.fc_t1.add(tf.keras.layers.Dense(diffusion_step_embed_dim_mid, input_shape=(diffusion_step_embed_dim_in,), activation=None)) 

        self.fc_t2 = tf.keras.models.Sequential()
        self.fc_t2.add(tf.keras.layers.Dense(diffusion_step_embed_dim_out, input_shape=(diffusion_step_embed_dim_mid,), activation=None)) 
        
        self.residual_blocks = []
        
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

            
    def call(self, input_data):
        noise, conditional, diffusion_steps = input_data

        diffusion_step_embed = calc_diffusion_step_embedding(diffusion_steps, self.diffusion_step_embed_dim_in)
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, conditional, diffusion_step_embed))  
            skip += skip_n  

        return skip * math.sqrt(1.0 / self.num_res_layers)  


class SSSDS4Imputer(tf.keras.layers.Layer):                                        #===========================
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

        self.init_conv = tf.keras.models.Sequential()
        #self.init_conv.add(tf.keras.layers.Flatten())
        self.init_conv.add(Conv(out_channels = res_channels, kernel_size=1))  # 激活函数relu已经嵌入了上面的Conv() class，所以这里不用写
        
        
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
        
        self.final_conv = tf.keras.models.Sequential()
        self.final_conv.add(Conv(out_channels = skip_channels, kernel_size=1))
        self.final_conv.add(ZeroConv1d(out_channel = out_channels))

    def call(self, input_data):
        
        noise, conditional, mask, diffusion_steps = input_data 

        conditional = conditional * mask
        conditional = tf.concat([conditional, mask], axis=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y
