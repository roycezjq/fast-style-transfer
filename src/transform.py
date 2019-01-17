import tensorflow as tf

WEIGHTS_INIT_STDEV = .1


def net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_transpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_transpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds


def _conv_layer(_net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(_net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    _net = tf.nn.conv2d(_net, weights_init, strides_shape, padding='SAME')
    _net = _instance_norm(_net)
    if relu:
        _net = tf.nn.relu(_net)

    return _net


def _conv_transpose_layer(_net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(_net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in _net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(_net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    _net = tf.nn.conv2d_transpose(_net, weights_init, tf_shape, strides_shape, padding='SAME')
    _net = _instance_norm(_net)
    return tf.nn.relu(_net)


def _residual_block(_net, filter_size=3):
    tmp = _conv_layer(_net, 128, filter_size, 1)
    return _net + _conv_layer(tmp, 128, filter_size, 1, relu=False)


def _instance_norm(_net, train=True):
    batch, rows, cols, channels = [i.value for i in _net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(_net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (_net - mu) / (sigma_sq + epsilon) ** .5
    return scale * normalized + shift


def _conv_init_vars(_net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in _net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init
