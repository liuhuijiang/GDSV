import tensorflow as tf
from tensorflow.python.ops import array_ops


def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


def accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def focal_loss(pred, y):
    gamma = 2
    alpha = 0.512
    pred = tf.nn.softmax(pred)
    zeros = array_ops.zeros_like(pred, dtype=pred.dtype)
    pos_p_sub = array_ops.where(y > zeros, y - pred, zeros)  # positive sample

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(y > zeros, zeros, pred)  # negative sample
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(pred, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - pred, 1e-8, 1.0))
    return tf.reduce_mean(tf.reduce_sum(per_entry_cross_ent, axis=1))


def dice_loss(preds, labels, smooth=1):  # DL
    inse = tf.reduce_sum(preds * labels, axis=1)
    l = tf.reduce_sum(preds * preds, axis=1)
    r = tf.reduce_sum(labels * labels, axis=1)
    dice = (2. * inse + smooth) / (l + r + smooth)
    loss = tf.reduce_mean(1 - dice)
    return loss


def DSC_loss(y_pred, y_true):  # https://www.cnblogs.com/hotsnow/p/10954624.html
    soomth = 0.5
    y_pred_rev = tf.subtract(1.0, y_pred)
    nominator = tf.multiply(tf.multiply(2.0, y_pred_rev), y_pred) * y_true
    denominator = tf.multiply(y_pred_rev, y_pred) + y_true
    dsc_coe = tf.divide(nominator, denominator)
    return tf.reduce_mean(dsc_coe)