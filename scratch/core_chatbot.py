import os
import base64
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")

if not API_KEY:
    raise SystemExit("API_KEY not found in environment variables. Please set it in your .env file.")

os.environ["GEMINI_API_KEY"] = API_KEY

client = genai.Client()

def text_completion_example():
    response = client.interactions.create(
        model="gemini-3.5-flash",
        input="Write a short poem about the ocean."
    )
    print("Text Completion Response:", response.output_text)

def multimodal_example(prompt, image_url=None, image_b64=None):
    if image_url is None and image_b64 is None:
        raise ValueError("Either image_url or image_b64 must be provided.")
    elif image_url is not None:
        response = client.interactions.create(
            model="gemini-3.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "image_url": image_url}
            ]
        )
    else:
        response = client.interactions.create(
            model="gemini-3.5-flash",
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "data": image_b64, "mime_type": "image/png"}
            ]
        )
    print("Multimodal Response:", response.output_text)

def multimodel_generation_example(prompt):
    interaction = client.interactions.create(
        model="gemini-3.1-flash-image",
        input="Generate an image of a futuristic city skyline at sunset",
    )
    with open("data/samples/output_image.png", "wb") as f:
        f.write(base64.b64decode(interaction.output_image.data))


def main():
    print("Running text completion example...")
    text_completion_example()

    # import base64
    # with open("data/samples/famous_person.png", "rb") as image_file:
        # image_data = image_file.read()

    # image_b64 = base64.b64encode(image_data).decode("utf-8")
    # print("Running multimodal example...")
    # text = "who is this person? The story of this person is very interesting. Can you tell me more about them?"
    # multimodal_example(text, image_b64=image_b64)

    # prompt="Generate an image of a futuristic underground city skyline at sunset",
    # multimodel_generation_example(prompt)
if __name__ == "__main__":
    main()