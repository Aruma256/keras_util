def stack(input, *nodes):
    x = input
    for node in nodes:
        if isinstance(node, list):
            x = [stack(x, *branch) for branch in node]
        else:
            x = node(x)
    return x

