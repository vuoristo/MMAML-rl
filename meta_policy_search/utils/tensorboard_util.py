import tensorflow as tf

def log_scalar(summary_writer, tag, value, step):
    """from
    https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    """
    summary = tf.Summary(
        value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, step)
