import torch

def get_batch(data, block_size, batch_size):

    indexes = torch.randint(
        len(data) - block_size, 
        (batch_size,)
    )

    x = torch.stack([data[i:i+block_size] for i in indexes])
    y = torch.stack([data[i+1:i+block_size+1] for i in indexes])

    return x, y
