# First, modify the train_model function to use our custom class:
def train_model(save_backbone=True):
    output_dir = '/home/data_science/geo_testing/COO_V3'
    backbone_dir = os.path.join(output_dir, 'pretrained_backbone')
    num_epochs = 40
    
    # Set up logging
    logging.basicConfig(
        filename=os.path.join(output_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    folder_path = '/home/data_science/geo_testing/COO_V3'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset, test_dataset, len_of_labels, label2id, id2label, all_labels, config = create_datasets(folder_path, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    # Initialize custom model instead of standard LayoutLMv2
    model = LayoutLMv2ForTokenClassificationCustom(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    global_step = 0
    best_loss = None
    best_precision = None
    best_recall = None
    best_f1 = None
    best_model_flag_high = False
    best_model_flag_low = False

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].long().to(device)
            bbox = batch['bbox'].long().to(device)
            image = batch['image'].float().to(device)
            attention_mask = batch['attention_mask'].long().to(device)
            token_type_ids = batch['token_type_ids'].long().to(device)
            labels = batch['labels'].long().to(device)

            optimizer.zero_grad()

            # Modified forward pass using custom model
            logits, sequence_output = model(
                input_ids=input_ids,
                bbox=bbox,
                image=image,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # Calculate loss manually since we're not using the standard forward pass
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, model.num_labels)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)

            # Rest of training logic remains the same
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Log training progress
            if (global_step + 1) % len(train_loader) == 0:
                print(f"Loss after {global_step} steps: {loss.item()}")
                train_writer.add_scalar("train loss", loss.detach(), epoch)

        # Evaluation loop
        model.eval()
        val_loss = 0.0
        preds_val = None
        out_label_ids = None
        
        for batch in tqdm(test_loader, desc="Evaluating"):
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                bbox = batch['bbox'].to(device)
                image = batch['image'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)

                # Modified forward pass for evaluation
                logits, _ = model(
                    input_ids=input_ids,
                    bbox=bbox,
                    image=image,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Calculate validation loss
                loss_fct = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, model.num_labels)
                active_labels = labels.view(-1)
                val_batch_loss = loss_fct(active_logits, active_labels)
                val_loss += val_batch_loss.item()

                if preds_val is None:
                    preds_val = logits.detach().cpu().numpy()
                    out_label_ids = labels.detach().cpu().numpy()
                else:
                    preds_val = np.append(preds_val, logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        # Calculate metrics and save best model
        val_result, class_report = results_test(preds_val, out_label_ids, list(set(all_labels)))
        
        # Save model logic
        if save_backbone and (best_model_flag_high or best_model_flag_low):
            # Save custom model backbone
            save_pretrained_backbone(model, backbone_dir)
            
            # Also save the classification head separately if needed
            classifier_state = {
                'classifier': model.classifier.state_dict(),
                'num_labels': model.num_labels
            }
            torch.save(classifier_state, os.path.join(backbone_dir, 'classifier.pth'))
            
    return model, backbone_dir

# Function to load the custom model
def load_custom_model(model_path, num_labels):
    """
    Load custom model with pretrained weights
    """
    config = LayoutLMv2Config.from_pretrained(model_path)
    config.num_labels = num_labels
    
    # Initialize custom model
    model = LayoutLMv2ForTokenClassificationCustom(config)
    
    # Load backbone weights
    backbone_state_dict = torch.load(os.path.join(model_path, 'backbone.pth'))
    model.load_state_dict(backbone_state_dict, strict=False)
    
    # Load classifier if available and if number of labels matches
    classifier_path = os.path.join(model_path, 'classifier.pth')
    if os.path.exists(classifier_path):
        classifier_state = torch.load(classifier_path)
        if classifier_state['num_labels'] == num_labels:
            model.classifier.load_state_dict(classifier_state['classifier'])
    
    return model