import torch
from transformers import LayoutLMv2PreTrainedModel, LayoutLMv2Model
from transformers import LayoutLMv2ForTokenClassification
from torch import nn

class LayoutLMv2ForTokenClassificationCustom(LayoutLMv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv2 = LayoutLMv2Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, *args, **kwargs):
        outputs = self.layoutlmv2(*args, **kwargs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits, sequence_output

def save_pretrained_backbone(model, save_path):
    """
    Save only the backbone (feature extractor) of the model
    
    Args:
        model: Trained LayoutLMv2 model
        save_path: Path to save the backbone
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save the backbone configuration
    model.layoutlmv2.config.save_pretrained(save_path)
    
    # Save only the backbone weights
    backbone_state_dict = {
        k: v for k, v in model.state_dict().items()
        if 'classifier' not in k
    }
    
    torch.save(backbone_state_dict, os.path.join(save_path, 'backbone.pth'))
    
    # Save additional training info
    training_info = {
        'hidden_size': model.config.hidden_size,
        'max_position_embeddings': model.config.max_position_embeddings,
        'max_2d_position_embeddings': model.config.max_2d_position_embeddings,
    }
    
    with open(os.path.join(save_path, 'training_info.json'), 'w') as f:
        json.dump(training_info, f)

def load_pretrained_backbone(model_path, num_labels):
    """
    Load pretrained backbone and initialize new classification head
    
    Args:
        model_path: Path to the saved backbone
        num_labels: Number of labels for the new task
        
    Returns:
        Initialized model with pretrained backbone
    """
    # Load configuration
    config = LayoutLMv2Config.from_pretrained(model_path)
    config.num_labels = num_labels
    
    # Create new model with updated number of labels
    model = LayoutLMv2ForTokenClassification(config)
    
    # Load pretrained backbone weights
    backbone_state_dict = torch.load(os.path.join(model_path, 'backbone.pth'))
    
    # Filter out classifier weights from the state dict
    model_state_dict = model.state_dict()
    for key in backbone_state_dict:
        if key in model_state_dict and 'classifier' not in key:
            model_state_dict[key] = backbone_state_dict[key]
    
    model.load_state_dict(model_state_dict, strict=False)
    return model

# Modified training function with backbone saving
def train_model(save_backbone=True):
    """Modified training function that saves the backbone separately"""
    output_dir = '/home/data_science/geo_testing/COO_V3'
    backbone_dir = os.path.join(output_dir, 'pretrained_backbone')
    num_epochs = 40
    
    # ... [rest of your existing training code] ...
    
    # After training, save the backbone separately if requested
    if save_backbone and (best_model_flag_high or best_model_flag_low):
        model_to_save = best_model_low if best_model_flag_low else model
        save_pretrained_backbone(model_to_save, backbone_dir)
        print(f"Saved pretrained backbone to {backbone_dir}")
    
    return model, backbone_dir

# Example usage for new dataset with different labels
def train_on_new_dataset(pretrained_backbone_path, new_num_labels):
    """
    Train on a new dataset using pretrained backbone
    
    Args:
        pretrained_backbone_path: Path to the pretrained backbone
        new_num_labels: Number of labels in the new dataset
    """
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load pretrained backbone with new classification head
    model = load_pretrained_backbone(pretrained_backbone_path, new_num_labels)
    
    # Freeze backbone layers (optional)
    for param in model.layoutlmv2.parameters():
        param.requires_grad = False
    
    # Only train the classification head
    trainable_params = model.classifier.parameters()
    optimizer = AdamW(trainable_params, lr=5e-5)
    
    # Create datasets
    folder_path = '/home/data_science/geo_testing/COO_V3'
    train_dataset, test_dataset, _, label2id, id2label, all_labels, _ = create_datasets(
        folder_path, 
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            # ... [Your existing training loop code] ...
            pass
    
    return model