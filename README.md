BLIP Based Image Classification with Hessian Sharpness Analysis

Project Description: Our project implements an image classification pipeline using a BLIP vision language model as a frozen feature extractor, paired with a linear classifier trained on top of BLIP image embeddings. Additionally, the project computes Hessian Sharpness (largest eigenvalue of the Hessian) during the training to analyze loss landscape flatness. The project also integrates BLIP captioning, enabling both classification and caption generation for inference.

Table of Content: 

- BLIPimageclassfication (v.1.).ipynb: The main file. After transferring the dataset files to the directory ALL in your google drive, lay out the files in the structure below

                         ___________
                        |    ALL    |
                        |___________|
                    __________|__________
                   |                     |
              ___________           ___________
             |    cat    |         |  non-cat  |
             |___________|         |___________|



- 
