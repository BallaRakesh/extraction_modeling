from mistral_common.protocol.instruct.messages import (
    UserMessage,
    TextChunk,
    ImageURLChunk,
    ImageChunk,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from PIL import Image

# Load the tokenizer
tokenizer = MistralTokenizer.from_model("pixtral")

# Load the local image
image_path = "local_image.jpg"  # Replace with your actual image path
image = Image.open(image_path).convert("RGB")

# Tokenize text and image
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="Describe this image"),
                    ImageChunk(image=image),
                ]
            )
        ],
        model="pixtral",
    )
)

# Extract tokenized data
tokens, text, images = tokenized.tokens, tokenized.text, tokenized.images

# Print token and image count
print("# tokens:", len(tokens))
print("# images:", len(images))