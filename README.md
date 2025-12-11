BLIP Based Image Classification with Hessian Sharpness Analysis

Project Description: Our project implements an image classification pipeline using a BLIP vision language model as a frozen feature extractor, paired with a linear classifier trained on top of BLIP image embeddings. Additionally, the project computes Hessian Sharpness (largest eigenvalue of the Hessian) during the training to analyze loss landscape flatness. The project also integrates BLIP captioning, enabling both classification and caption generation for inference.

Table of Content: 

- ALL_SPLIT: The main dataset file. After transferring the dataset files to the directory in your google drive, lay out is divided into training(70%), validation(15%) and test(15%).
- Final version BLIP deep learning project .ipynb - The final code

  
References
Hugging face BLIP model - The BLIP model used in final code- by Salesforce https://huggingface.co/Salesforce/blip-image-captioning-base

Downloaded only 100 images out of the 25,000 Cat Images by Tamilselvan Arjunan https://www.kaggle.com/datasets/tamilselvanarjunan/image1?resource=download
Downloaded only 100 random images from here https://www.kaggle.com/datasets/lprdosmil/unsplash-random-images-collection?resource=download

