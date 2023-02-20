from io import BytesIO
from typing import List

import requests
from PIL import Image

original_api = "https://pixabay.com/api/?key="
no_to_retrieve = 5
pixabay_api_key = "XXXXXXXXXXXXXXX"  # API key


def scrape_images(words_to_search=["Giraffe", "Tiger", "Fruits"]) -> List:
    all_images = []

    for pixabay_search_keyword in words_to_search:

        pixabay_api = (
            original_api
            + pixabay_api_key
            + "&q="
            + pixabay_search_keyword.lower()
            + "&image_type=photo&safesearch=true&per_page="
            + str(no_to_retrieve)
        )
        response = requests.get(pixabay_api)
        output = response.json()

        for each in output["hits"]:
            imageurl = each["webformatURL"]
            response = requests.get(imageurl)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            all_images.append(image)

    return all_images


if __name__ == "__main__":
    images = scrape_images()
    print(len(images), images[0])
