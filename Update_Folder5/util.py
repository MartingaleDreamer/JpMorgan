import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import torch
import random


def flatten(v):
    """
    Flatten a list of lists/tuples
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path
    
    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 13:         # 这里在检查文件路径下的文件夹每个文件名字的长度，如果文件名长度都不够4个(或等于4个)字符则跳出
            continue
        if f[-13:] == '.weight.index':      # 这里检查在所有文件名长度大于4个字符的文件中，文件名末尾7个是否是.weight，即checkpoint文件，如果是，测查看epoch数值大小(和-1比大小取更大值)，E.g. 100000.pkl，并最终返回
            try:
                epoch = max(epoch, int(f[:-13]))
            except:
                continue
    return epoch


def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print("{} Parameters: {:.6f}M".format(
            net.__class__.__name__, params / 1e6), flush=True)


# Utilities for diffusion models

def std_normal(size):
    """
    Generate the standard Gaussian variable of a certain size
    """

    return tf.random.normal(mean=0, stddev=1, shape=size)


def calc_diffusion_step_embedding(diffusion_steps, diffusion_step_embed_dim_in):
    """
    Embed a diffusion step $t$ into a higher dimensional space
    E.g. the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ... , sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ... , cos(t * 10^(63*4/63))]

    Parameters:
    diffusion_steps (torch.long tensor, shape=(batchsize, 1)):     
                                diffusion steps for batch data
    diffusion_step_embed_dim_in (int, default=128):  
                                dimensionality of the embedding space for discrete diffusion steps
    
    Returns:
    the embedding vectors (torch.tensor, shape=(batchsize, diffusion_step_embed_dim_in)):
    """

    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = tf.math.exp(torch.arange(half_dim) * -_embed)
    _embed = tf.cast(_embed, dtype=tf.int32)  # Add
    _embed = diffusion_steps * _embed
    _embed = tf.cast(_embed, dtype=tf.float32)  # Add
    diffusion_step_embed = tf.concat((tf.math.sin(_embed),
                                      tf.math.cos(_embed)), 1)

    return diffusion_step_embed


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule                       # [Zihao]: torch used
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
                1 - Alpha_bar[t])  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1})
        # / (1-\bar{\alpha}_t)
    Sigma = tf.math.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t                     # [Zihao]: torch used

    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma # [Zihao]: 将计算出的参数分配到_dh字典内，对应键值对为左侧式子
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def sampling(net, size, diffusion_hyperparams, cond, mask, only_generate_missing=0, guidance_weight=0):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors 
    
    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = std_normal(size)

    for t in range(T - 1, -1, -1):
        if only_generate_missing == 1:
            x = x * (1 - mask) + cond * mask
        diffusion_steps = t * tf.ones([size[0], 1],tf.float32)  # use the corresponding reverse step
        epsilon_theta = net((x, cond, mask, diffusion_steps))  # predict \epsilon according to \epsilon_\theta
        epsilon_theta = torch.tensor(epsilon_theta.numpy())
        # update x_{t-1} to \mu_\theta(x_t)

        #print('This is x shape: ',x.shape)
        #print('This is Alpha shape: ',Alpha.shape)
        #print('This is Alpha_bar shape: ',Alpha_bar.shape)
        #print('This is epsilon_theta shape: ',epsilon_theta.shape)

        x = (x - (1 - Alpha[t]) / tf.math.sqrt(1 - Alpha_bar[t]) * epsilon_theta) / tf.math.sqrt(Alpha[t])
        if t > 0:
            x = x + Sigma[t] * std_normal(size)  # add the variance term to x_{t-1}

    return x


def training_loss(net, loss_fn, X, diffusion_hyperparams, only_generate_missing=1):
    """
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch network):            the wavenet model
    loss_fn (torch loss function):  the loss function, default is nn.MSELoss()
    X (torch.tensor):               training data, shape=(batchsize, 1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors       
    
    Returns:
    training loss
    """

    _dh = diffusion_hyperparams     # 计算出的扩散模型超参数
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio = X[0]
    cond = X[1]
    mask = X[2]
    loss_mask = X[3]

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = tf.random.uniform(minval=0, maxval=T, shape=(B, 1, 1), dtype=tf.int32)  # randomly sample diffusion steps from 1~T (random integers)
    diffusion_steps = diffusion_steps.numpy()
    z = std_normal(audio.shape)
    if only_generate_missing == 1:
        z = audio * mask + z * (1 - mask)

    #print(Alpha_bar)
    
    transformed_X = tf.math.sqrt(Alpha_bar[diffusion_steps]) * audio + tf.math.sqrt(1 - Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    diffusion_steps = tf.convert_to_tensor(diffusion_steps,dtype=tf.int32)

    print(transformed_X.shape)
    
    epsilon_theta = net((transformed_X, cond, mask, tf.reshape(diffusion_steps,[B, 1])))  # predict \epsilon according to \epsilon_\theta

    #print(z.shape)  # Add
    #print(loss_mask.shape)
    #print(epsilon_theta.shape)
    #print(loss_mask)
    #print(len(z))

    z = torch.tensor(z.numpy())  # 这里转为torch为了下面的loss_fn(z[loss_mask],epsilon_theta[loss_mask])
    loss_mask = torch.tensor(loss_mask.numpy(),dtype=torch.bool)
    epsilon_theta = torch.tensor(epsilon_theta.numpy())
    #print('这是输出: ',epsilon_theta,'结尾--------')
    #print('查看真实值Size: ',z.size(),'结尾==============')
    #print('查看真实值Size: ',z[loss_mask].size(),'结尾++++++++++')
    #print('查看模型预测值Size: ',epsilon_theta[loss_mask].size(),'结尾________')

    if only_generate_missing == 1:
        return loss_fn(tf.convert_to_tensor(z[loss_mask].numpy()),tf.convert_to_tensor(epsilon_theta[loss_mask].numpy())), tf.convert_to_tensor(z.numpy())      # only_generate_missing = 1时候，只使用样本中missing部分进行diffusion获得的MSE值,这里的loss_mask是取反mask，原本mask中0表示missing, 现在loss_mask中1表示missing，于是这里保留了missing部分，epsilon_theta[loss_mask]是基于扩散模型计算出的值，z[loss_mask]是target目标值，即来自training_set的batch真实值，此函数返回两者的MSE和真实的y_train
    elif only_generate_missing == 0:
        return loss_fn(tf.convert_to_tensor(z.numpy()),tf.convert_to_tensor(epsilon_theta.numpy())), tf.convert_to_tensor(z.numpy())                 # only_generate_missing = 0时候，只使用全样本进行diffusion获得的MSE值，epsilon_theta是基于扩散模型计算出的预测值, z是来自training_set的batch真实值，此函数返回两者的MSE和真实的y_train


def get_mask_rm(sample, k):     # 这是random missing的情况，表示所有列(features)都保留了数据，但是在每一个feature下的不同时间点，会有随机的missing时间点
    """Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""

    mask = tf.ones(sample.shape).numpy()                                       #[Zihao]: torch used 产生都是数值1的torch tensor, 几行几列取决于sample.shape
    length_index = tf.convert_to_tensor(range(mask.shape[0]))  # lenght of series indexes  # 序列长度，表示数据矩阵有多少行，即有多少个时间点
    for channel in range(mask.shape[1]):        # 这是对于数据矩阵的每一列，即对于每一个feature
        perm = tf.random.shuffle(length_index)  # 产生随机排列，从0(包括，即第一个时间点)到n-1(包括，即最后一个时间点)
        idx = perm[0:k].numpy()               # 这里的k是 number of missing data points, 比如在SSSDS4中设定为90个missing时间点，我们选取前90个随机数，表示这些被选中的数据编号变为missing data point，每一列都会产生不同的随机missing data points
        mask[:, channel][idx] = 0         # 将每一列中随机选中的missing points设置值为0，表示missing，其他的继续保留为1，这就是mask的由来，0表示missing的点，1表示保留的点，与sample所有目标值相乘后，则会失去一些与0相乘的值
    mask = tf.convert_to_tensor(mask)
    return mask                     # 返回的mask也是一个torch tensor


def get_mask_mnr(sample, k):                # 这里的segments指的就是连续的一小段时间，segments是随机的，但是一旦segment确定，里面的时间点确定，不再是随机的
    """Get mask of random segments (non-missing at random) across channels based on k,  
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to preserved
    as per ts imputers"""
                                                     
    mask = torch.ones(sample.shape)                                       #[Zihao]: torch used 产生都是数值1的torch tensor, 几行几列取决于sample.shape
    length_index = torch.tensor(range(mask.shape[0]))  #还是表示有多少行，即有多少时间点
    list_of_segments_index = torch.split(length_index, k)  # 这里是将一个torch tensor进行分割，分割成多个chunk(块，每一块都是一个独立tensor),每一块内有k个长度的元素(segments，即时间点)
    for channel in range(mask.shape[1]):
        s_nan = random.choice(list_of_segments_index)  # 这里表示随机选择了一个chunk，即随机选择了一个tensor， 每一个tensor里面包含的是一小段的连续时间点
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0  # 在每一个feature下，都会随机选择这么一小段tensor，即一小段的连续时间点，设定值为0,每一个feature下选择的一小段时间点都不同
                                  # 与sample矩阵相乘后每一个feature内都会有一小段，连续k个时间点missing

    return mask                     # 返回的mask也是一个torch tensor


def get_mask_bm(sample, k):
    """Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed, and 1's to be preserved
    as per ts imputers"""

    mask = torch.ones(sample.shape)                                      #[Zihao]: torch used 产生都是数值1的torch tensor, 几行几列取决于sample.shape
    length_index = torch.tensor(range(mask.shape[0]))  #还是表示有多少行，即有多少时间点
    list_of_segments_index = torch.split(length_index, k)  # 这里是将一个torch tensor进行分割，分割成多个chunk(块，每一块都是一个独立tensor),每一块内有k个长度的元素(segments，即时间点)
    s_nan = random.choice(list_of_segments_index)  # 这里随机选择了一个segment，即一小段连续时间点，选择好后不再改变，带入下面的每一个feature中
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0]:s_nan[-1] + 1] = 0  # 在每一个feature下，都是这相同的一段时间内的连续时间点missing了，所有的feature下都是这一段时间，每一个feature下都相同，所以这叫做bm情况

    return mask                     # 返回的mask也是一个torch tensor
