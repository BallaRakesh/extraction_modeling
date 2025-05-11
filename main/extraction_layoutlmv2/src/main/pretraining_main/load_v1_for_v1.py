import torch
import torch.nn as nn
from transformers import LayoutLMForTokenClassification, LayoutLMv2Model



import torch
import torch.nn as nn
from transformers import LayoutLMForTokenClassification, LayoutLMv2Model

def load_layoutlm_v1_features_into_v2_temp(layoutlmv2_model: LayoutLMv2Model) -> LayoutLMv2Model:
    """
    Loads language and spatial features from LayoutLMv1 into LayoutLMv2 model,
    accounting for different embedding dimensions and concatenation strategy.
    
    Args:
        layoutlmv2_model: A pretrained LayoutLMv2 model to load the features into
        
    Returns:
        LayoutLMv2Model with loaded v1 features
    """
    # Load LayoutLMv1 model
    layoutlm_v1 = LayoutLMForTokenClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased", 
        num_labels=1
    )
    
    # Transfer word embeddings (same dimension in both models)
    layoutlmv2_model.embeddings.word_embeddings.weight.data = \
        layoutlm_v1.layoutlm.embeddings.word_embeddings.weight.data.clone()
    
    # Transfer position embeddings (same dimension in both models)
    min_length = min(
        layoutlm_v1.layoutlm.embeddings.position_embeddings.weight.data.size(0),
        layoutlmv2_model.embeddings.position_embeddings.weight.data.size(0)
    )
    layoutlmv2_model.embeddings.position_embeddings.weight.data[:min_length] = \
        layoutlm_v1.layoutlm.embeddings.position_embeddings.weight.data[:min_length].clone()
    
    # Transfer spatial embeddings with dimension reduction
    # For x and y coordinates (reducing from 768 to 128 dimensions)
    def reduce_embedding_dim(v1_embedding, v2_embedding):
        """Reduces embedding dimension using average pooling"""
        v1_weight = v1_embedding.weight.data
        pool_size = v1_weight.size(1) // v2_embedding.weight.data.size(1)
        reduced = torch.nn.functional.avg_pool1d(
            v1_weight.transpose(0, 1).unsqueeze(0),
            kernel_size=pool_size,
            stride=pool_size
        ).squeeze(0).transpose(0, 1)
        return reduced

    # Transfer x position embeddings (reduce from 768 to coordinate_size)
    v2_x_emb = reduce_embedding_dim(
        layoutlm_v1.layoutlm.embeddings.x_position_embeddings,
        layoutlmv2_model.embeddings.x_position_embeddings
    )
    layoutlmv2_model.embeddings.x_position_embeddings.weight.data = v2_x_emb

    # Transfer y position embeddings (reduce from 768 to coordinate_size)
    v2_y_emb = reduce_embedding_dim(
        layoutlm_v1.layoutlm.embeddings.y_position_embeddings,
        layoutlmv2_model.embeddings.y_position_embeddings
    )
    layoutlmv2_model.embeddings.y_position_embeddings.weight.data = v2_y_emb

    # Transfer h and w embeddings (reduce from 768 to shape_size)
    v2_h_emb = reduce_embedding_dim(
        layoutlm_v1.layoutlm.embeddings.h_position_embeddings,
        layoutlmv2_model.embeddings.h_position_embeddings
    )
    layoutlmv2_model.embeddings.h_position_embeddings.weight.data = v2_h_emb

    v2_w_emb = reduce_embedding_dim(
        layoutlm_v1.layoutlm.embeddings.w_position_embeddings,
        layoutlmv2_model.embeddings.w_position_embeddings
    )
    layoutlmv2_model.embeddings.w_position_embeddings.weight.data = v2_w_emb
        
    return layoutlmv2_model





def load_layoutlm_v1_features_into_v2(layoutlmv2_model: LayoutLMv2Model) -> LayoutLMv2Model:
    """
    Loads language and spatial features from LayoutLMv1 into LayoutLMv2 model.
    
    Args:
        layoutlmv2_model: A pretrained LayoutLMv2 model to load the features into
        
    Returns:
        LayoutLMv2Model with loaded v1 features
    """
    # Load LayoutLMv1 model
    layoutlm_v1 = LayoutLMForTokenClassification.from_pretrained(
        "microsoft/layoutlm-base-uncased", 
        num_labels=1
    )
    
    # Transfer word embeddings
    layoutlmv2_model.embeddings.word_embeddings.weight.data = \
        layoutlm_v1.layoutlm.embeddings.word_embeddings.weight.data.clone()
    
    # Transfer position embeddings
    # Note: LayoutLMv2 might have different max_position_embeddings
    v1_pos_emb = layoutlm_v1.layoutlm.embeddings.position_embeddings.weight.data
    v2_pos_emb = layoutlmv2_model.embeddings.position_embeddings.weight.data
    
    # Copy up to the minimum length between v1 and v2
    min_length = min(v1_pos_emb.size(0), v2_pos_emb.size(0))
    layoutlmv2_model.embeddings.position_embeddings.weight.data[:min_length] = \
        v1_pos_emb[:min_length].clone()
    
    # Transfer spatial embeddings
    # Layout LMv1 uses x1, y1, x2, y2, h, w spatial embeddings
    # Layout LMv2 uses only x1, y1, x2, y2
    v1_spatial_embeddings = layoutlm_v1.layoutlm.embeddings.x_position_embeddings.weight.data
    
    # Initialize x coordinate embeddings
    layoutlmv2_model.embeddings.x_position_embeddings.weight.data = \
        v1_spatial_embeddings.clone()
    
    v1_spatial_embeddings = layoutlm_v1.layoutlm.embeddings.y_position_embeddings.weight.data
    
    # Initialize y coordinate embeddings
    layoutlmv2_model.embeddings.y_position_embeddings.weight.data = \
        v1_spatial_embeddings.clone()
        
    return layoutlmv2_model

# Usage example:
def load_and_initialize_layoutlmv2_with_v1_features():
    """
    Creates and initializes a LayoutLMv2 model with LayoutLMv1 features
    
    Returns:
        Initialized LayoutLMv2 model
    """
    # Initialize LayoutLMv2 with random weights
    layoutlmv2_model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")
    
    # Load LayoutLMv1 features
    layoutlmv2_model = load_layoutlm_v1_features_into_v2(layoutlmv2_model)
    
    return layoutlmv2_model


# Initialize model with v1 features
model = load_and_initialize_layoutlmv2_with_v1_features()
# Or if you already have a v2 model:
model = load_layoutlm_v1_features_into_v2(your_existing_v2_model)