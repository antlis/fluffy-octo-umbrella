# from duckduckgo_search import DDGS
# from fastcore.all import *

from fastai.vision.all import *

path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper() 

dls = ImageDataLoaders.from_name_func('.',
    get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat,
    item_tfms=Resize(192))

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

learn.export('model.pkl')

# def search_images(term, max_images=30):
#     print(f"Searching for '{term}'")
#     with DDGS() as ddgs:
#         # generator which yields dicts with:
#         # {'title','image','thumbnail','url','height','width','source'}
#         search_results = ddgs.images(keywords=term)       
#         # grap number of max_images urls
#         image_urls = [next(search_results).get("image") for _ in range(max_images)]
#         # convert to L (functionally extended list class from fastai)
#         return L(image_urls)

# # example usage:
# urls = search_images("dog images", max_images=10)
# print(urls[0])
