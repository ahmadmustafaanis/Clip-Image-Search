import clip
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from scrapper import scrape_images

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def image_text_search(image: Image, text_descriptions: list):
    text = clip.tokenize(text_descriptions).to(device)
    image = preprocess(image).unsqueeze(0).to(device)  # Preprocess the Image

    with torch.no_grad():
        logits_per_image, logits_per_text = model(
            image, text
        )  # Pass both text and Image as Input to the model
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    results = dict(zip(text_descriptions, map(lambda x: x * 100, probs[0])))
    results = {
        k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)
    }  # Sorted Results

    return results


if __name__ == "__main__":
    descriptions = [
        "A Pictrue of a Tiger and Girl on Rocks",
        "A picture of Donkey and a Man",
        "A picture of a red car",
        "A picture of a Sparrow and Butterfly",
        "A picture of Animal and Human",
    ]

    image = cv2.imread("data/tiger_girls_query.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    results = image_text_search(image, descriptions)

    plt.imshow(np.array(image))
    title_string = ""
    for key, value in results.items():
        title_string += f"\n String: {key}: Similarity: {value:.2f}%"
    plt.ylabel(title_string)
    plt.show()
