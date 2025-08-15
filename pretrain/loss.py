import torch
import torch.nn.functional as F


def masked_language_modeling_loss(logits, labels, attention_mask, weight=None):
    """
    Calculate the loss for masked language modeling.

    Args:
        logits (torch.Tensor): The logits output by the model, with shape (batch_size, seq_length, vocab_size).
        labels (torch.Tensor): The masked labels, with shape (batch_size, seq_length).
                               Unmasked positions should be set to -100.
        attention_mask (torch.Tensor): The attention mask, with shape (batch_size, seq_length).
        weight (torch.Tensor, optional): Optional weights for weighting the loss, with shape (batch_size, seq_length).

    Returns:
        torch.Tensor: The calculated MLM loss.
    """

    # Use CrossEntropyLoss, ignoring positions where the label is -100
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    # Flatten logits and labels to fit the loss function's input requirements
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

    # Reshape the loss to (batch_size, seq_length)
    loss = loss.view(labels.size())

    # Keep only the loss for valid positions based on the attention mask
    loss = loss * attention_mask

    if weight is not None:
        # If weights are provided, apply them to the loss
        loss = loss * weight

    # Calculate the average loss, avoiding division by zero
    loss = loss.sum() / (attention_mask.sum() + 1e-8)

    return loss


def contrastive_loss(features1, features2, temperature=0.07):
    """
    Calculate the contrastive learning loss.

    Args:
        features1 (torch.Tensor): The first set of feature vectors, with shape (batch_size, hidden_dim).
        features2 (torch.Tensor): The second set of feature vectors, with shape (batch_size, hidden_dim).
        temperature (float): The temperature coefficient for scaling similarity.

    Returns:
        torch.Tensor: The calculated contrastive loss.
    """
    # Normalize feature vectors to be on the unit sphere
    features1 = F.normalize(features1, dim=1)  # Shape: (batch_size, hidden_dim)
    features2 = F.normalize(features2, dim=1)  # Shape: (batch_size, hidden_dim)

    # Calculate the similarity matrix (batch_size, batch_size)
    similarity_matrix = torch.matmul(features1, features2.T)

    # Scale the similarity matrix
    similarity_matrix = similarity_matrix / temperature

    # Create labels, positive samples correspond to the diagonal
    labels = torch.arange(features1.size(0)).to(features1.device)

    # Use CrossEntropyLoss
    loss_fct = torch.nn.CrossEntropyLoss()

    # Calculate the loss from features1 to features2
    loss1 = loss_fct(similarity_matrix, labels)

    # Calculate the loss from features2 to features1
    loss2 = loss_fct(similarity_matrix.T, labels)

    # Average the losses from both directions
    loss = (loss1 + loss2) / 2

    return loss
