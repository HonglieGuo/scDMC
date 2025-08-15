import torch
import torch.utils.checkpoint
from transformers.utils import logging

# Import necessary components from the geneformer library
from geneformer import DataCollatorForCellClassification

logger = logging.get_logger(__name__)

class GHLDataCollatorForCellClassification(DataCollatorForCellClassification):
    """
    Inherits from geneformer's DataCollatorForCellClassification.
    The main modification is in the _prepare_batch method to ensure compatibility
    with the custom evaluation process, and to add token_type_ids.
    It is assumed that the geneformer library is correctly installed and importable.
    """
    def __init__(self, tokenizer=None, *args, **kwargs):
        """
        Args:
            tokenizer: An instantiated Tokenizer object. Although we receive it, we do not pass it to super().__init__
                       to avoid conflicts with geneformer's internal tokenizer. The Trainer will still use it.
        """
        kwargs.pop('tokenizer', None)
        super().__init__(*args, **kwargs) # Note: tokenizer=tokenizer is no longer passed here

    def _prepare_batch(self, features):
        """
        Prepare batch data.
        Modifications:
        1. Adjust the features passed to the parent class's `_prepare_batch` to ensure it includes the 'label' key needed by the parent's internal logic.
        2. After parent class processing, add all-zero `token_type_ids`.
        3. Finally, add our own processed `labels` tensor with the correct dtype.
        """

        expected_keys_for_parent = ['input_ids', 'attention_mask', 'label']

        final_labels_tensor = None
        if "label" in features[0] and features[0]["label"] is not None:
            first = features[0]
            label_value = first["label"] # Get the label value of the first sample

            # Determine the type of the label value (handle Tensor, list, or scalar)
            if isinstance(label_value, torch.Tensor):
                # Get scalar value from tensor to determine type
                label_example = label_value.item() if label_value.numel() == 1 else label_value[0].item()
            elif isinstance(label_value, list):
                 # Get the first element from the list (if it exists) to determine the type
                label_example = label_value[0] if label_value else 0 # Use 0 as the default type determination value for an empty list
            else:
                # Use the label value directly
                label_example = label_value

            # Determine dtype based on sample type
            label_dtype = torch.long if isinstance(label_example, int) else torch.float

            try:
                # Create the final labels tensor we will use, with the correct type
                final_labels_tensor = torch.tensor([f["label"] for f in features], dtype=label_dtype)
            except Exception as e:
                logger.error(f"Error creating final_labels_tensor: {e}")
                final_labels_tensor = None # Set to None on failure

        features_for_parent = []
        for f in features:
            # Keep all keys required by the parent class
            parent_f = {k: f[k] for k in expected_keys_for_parent if k in f}
            features_for_parent.append(parent_f)

        try:
            batch = super()._prepare_batch(features_for_parent)
        except Exception as e:
            logger.error(f"Error calling parent's _prepare_batch: {e}")
            logger.error(f"Feature sample passed to parent (first one): {features_for_parent[0] if features_for_parent else 'None'}")
            raise e

        if 'input_ids' in batch and 'token_type_ids' not in batch: # Check if the parent class has already added it
            if isinstance(batch.get('input_ids'), torch.Tensor):
                batch['token_type_ids'] = torch.zeros_like(batch['input_ids'])
            else:
                 logger.warning("Could not generate 'token_type_ids' because 'input_ids' is missing or not a tensor in the batch.")

        if final_labels_tensor is not None:
            batch["labels"] = final_labels_tensor
        elif "labels" not in batch and "label" in features[0]: # If we failed and the parent also removed labels
             logger.warning("'labels' is missing from the final batch, although 'label' was present in the original features.")
        elif "label" not in features[0]:
             logger.debug("'label' was not in the original features, and 'labels' is not in the final batch.")

        if 'label' in batch:
            del batch['label']

        return batch