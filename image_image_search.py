from typing import List

import clip
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image

from scrapper import scrape_images

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


image_database = scrape_images()
image_database_processed = [
    preprocess(im) for im in image_database
]  # preprocess each Image
with torch.no_grad():
    database_embeddings = model.encode_image(
        torch.stack(image_database_processed)
    )  # Torch.Stack will help us to levragebatch processing to speed up the calculation


def image_image_search(query_image: Image, database_embeddings: torch.Tensor):
    query_embeddings = model.encode_image(
        preprocess(query_image).unsqueeze(0).to(device)
    )

    similariries = query_embeddings @ database_embeddings.T
    return similariries


if __name__ == "__main__":
    image_np = cv2.imread("data/query_apple.jpeg")
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_np)
    sim = image_image_search(image, database_embeddings)
    sim_dict = dict(
        zip(range(len(sim[0])), sim[0])
    )  # Use Dictionary to Sort the Results
    sorted_sim = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    top_sim = sorted_sim[:6]  # Get top 6 results

    fig, axs = plt.subplots(3, 3, figsize=(15, 6), facecolor="w", edgecolor="k")
    fig.subplots_adjust(hspace=0.5, wspace=0.001)
    plt.title("Image - Image Search Results")

    axs = axs.ravel()
    axs[0].imshow(image_np)
    axs[0].set_title("Query Image")
    axs[0].axes.xaxis.set_ticklabels([])
    axs[0].axes.yaxis.set_ticklabels([])

    for num, i in enumerate(top_sim, start=1):
        axs[num].imshow(image_database[i[0]])
        axs[num].set_title(f"Similarity: {i[1]:.2f}")
        axs[num].axes.xaxis.set_ticklabels([])
        axs[num].axes.yaxis.set_ticklabels([])

    fig.delaxes(axs[-1])
    fig.delaxes(axs[-2])

    plt.show()
