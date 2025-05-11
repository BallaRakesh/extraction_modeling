def infer_vision_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=False)
    vision_model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",   
                                                                    #  torch_dtype=torch.float16,         # Reduces memory usage
                                                                     load_in_4bit=True,                # Efficient memory optimization
                                                                     offload_folder="offload",
                                                                     device_map=device)
    all_output_list = []
    for batch in tqdm(dataset.select(range(300)), desc=f"Epoch {epoch+1}"):
        images = batch["messages"][0]
        labels = batch["messages"][1]
        current_image = images["content"][1]["image"]

        input_text = processor.apply_chat_template([images], tokenize=False, add_generation_prompt=True)
        
        input_img_label = processor(
        images = current_image,
        text = [input_text],
        return_tensors="pt"
        ).to("cuda")

        model_output = vision_model.generate(**input_img_label, max_length=800)
        model_output_text = processor.batch_decode(model_output, skip_special_tokens=True)
        all_output_list.append(model_output_text)

    return all_output_list