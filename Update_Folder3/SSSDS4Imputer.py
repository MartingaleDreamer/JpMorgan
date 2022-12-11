import math
#----------------------------------------------------------------------------------
import tensorflow as tf
import tensorflow_probability as tfp
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#-----------------------------------------------------------------------------------
from utils.util import calc_diffusion_step_embedding
from imputers.S4Model import S4Layer


def swish(x):
    return x * tf.math.sigmoid(x)


class Conv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = tf.keras.layers.Conv1D(data_format='channels_last', filters=out_channels, kernel_size=kernel_size, dilation_rate=dilation, padding=self.padding, input_shape = x[3:])(x) # padding="same"
        self.conv = tfp.layers.weight_norm.WeightNorm(self.conv)(x)
        tf.keras.initializers.HeNormal(self.conv.weight)

    def call(self, x):
        out = self.conv(x)
        return out
    
    
class ZeroConv1d(tf.keras.layers.Layer):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = tf.keras.layers.Conv1D(data_format='channels_last', filters=out_channel, kernel_size=1, padding="same", input_shape = x[3:])(x)
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
 
        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.S42 = S4Layer(features=2*self.res_channels,                             #********
                          lmax=s4_lmax,
                          N=s4_d_state,
                          dropout=s4_dropout,
                          bidirectional=s4_bidirectional,
                          layer_norm=s4_layernorm)
        
        self.cond_conv = Conv(2*in_channels, 2*self.res_channels, kernel_size=1)               #********

        self.res_conv = tf.keras.layers.Conv1D(data_format='channels_last', filters=res_channels, kernel_size=1)(input_data)
        self.res_conv = tfp.layers.weight_norm.WeightNorm(self.res_conv)
        tf.keras.initializers.HeNormal(self.res_conv.weight)

        
        self.skip_conv = tf.keras.layers.Conv1D(data_format='channels_last', filters=skip_channels, kernel_size=1)(input_data)
        self.skip_conv = tfp.layers.weight_norm.WeightNorm(self.skip_conv)
        tf.keras.initializers.HeNormal(self.skip_conv.weight)

    def call(self, input_data):                #----------------
        x, cond, diffusion_step_embed = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels                      
                 
        part_t = self.fc_t(diffusion_step_embed)
        part_t = part_t.view([B, self.res_channels, 1])  
        h = h + part_t
        
        h = self.conv_layer(h)
        h = self.S41(h.permute(2,0,1)).permute(1,2,0)     
        
        assert cond is not None
        cond = self.cond_conv(cond)
        h += cond
        
        h = self.S42(h.permute(2,0,1)).permute(1,2,0)
        
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

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

        self.fc_t1 = nn.Linear(diffusion_step_embed_dim_in, diffusion_step_embed_dim_mid)
        self.fc_t2 = nn.Linear(diffusion_step_embed_dim_mid, diffusion_step_embed_dim_out)
        
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels, 
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

            
    def forward(self, input_data):
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


class SSSDS4Imputer(tf.keras.layers.Layer):
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

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())
        
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
        
        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))

    def forward(self, input_data):
        
        noise, conditional, mask, diffusion_steps = input_data 

        conditional = conditional * mask
        conditional = torch.cat([conditional, mask.float()], dim=1)

        x = noise
        x = self.init_conv(x)
        x = self.residual_layer((x, conditional, diffusion_steps))
        y = self.final_conv(x)

        return y
