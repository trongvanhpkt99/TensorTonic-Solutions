def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    output=[]
    if tokens == []: return tokens
    if chunk_size >= len(tokens):
        return [tokens]
    for i in range(int(len(tokens)/(chunk_size-overlap))):
        head=i*(chunk_size-overlap)
        if len(tokens)-head>=chunk_size:
            output.append(tokens[head:head+chunk_size])
    return output
