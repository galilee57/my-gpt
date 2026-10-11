import torch

def get_batch(
        train_data, 
        val_data,
        split,
        block_size, 
        batch_size,
        device):

    # la fonction get_batch permet de générer un batch d'exemples pour l'entraînement ou la validation.
    data = train_data if split == 'train' else val_data

    indexes = torch.randint(
        len(data) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([data[i:i+block_size] for i in indexes])
    y = torch.stack([data[i+1:i+block_size+1] for i in indexes])

    return x.to(device), y.to(device)