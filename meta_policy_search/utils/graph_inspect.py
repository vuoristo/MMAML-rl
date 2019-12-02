import tensorflow as tf


def parents(op):
    return set(input.op for input in op.inputs)


def children(op):
    return set(op for out in op.outputs for op in out.consumers())


def get_graph():
    """Creates dictionary {node: {child1, child2, ..},..} for current
    TensorFlow graph. Result is compatible with networkx/toposort"""

    ops = tf.get_default_graph().get_operations()
    return {op: children(op) for op in ops}


def print_tf_graph(graph):
    """Prints tensorflow graph in dictionary form."""
    for node in graph:
        for child in graph[node]:
            print("%s -> %s" % (node.name, child.name))