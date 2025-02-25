# Toy-Object-Verification

**Objective:**  
Train a classifier designed to take two images as input and predict whether the shapes depicted in them are identical.
![Images Examples](images/samples.png)

### 1. Data Details

Each image is 32x32 pixels and contains either a circle or a square with random size, position, and rotation angle. The training set includes 5,000 images of squares and 5,000 images of circles to enable evaluation on unseen data, while the validation set has 1,000 images of each. For efficiency, only the parameters for sample generation are stored rather than the actual images. The target is represented by a binary label: **1** if the shapes in the pair are identical, and **0** if they differ.

### 2. Proposed Approaches  
| Approach             | Combination / Merging Method                                                    | Loss Function                                          |
|-------------------------------------|---------------------------------------------------------------------------------|--------------------------------------------------------|
| **Siamese NN:** Train Embeddings to Optimize Distance.   | Compute distance between feature vectors using one of the following metrics: <br> - Cosine similarity; <br> - L2 norm; <br> - L1 norm. <br> Train the model to have low diatance for images with the same objects, and the bigger distance for different objects. | - Contrastive Loss; <br> - Triplet Loss. |
| **Vector Merging:** One model is used to obtain embeddings for each image. Then, these embeddings are merged into a single vector and passed through a fully connected neural network (FNN) to predict the label.  | **Method A:** Concatenation                                                     | BCE                             |
|                                     | **Method B:** Element-wise multiplication                                      |BCE                             |
|                                     | **Method C:** Element-wise difference or sum                                   | BCE                             |
|                                     | **Method D:** Bilinear pooling (Bilinear Pooling)                              | BCE                             |
| **Cross-Attention:**  Apply cross-attention to the feature maps so that features from one image guide the selection of features from the other. Afterwards, the features are merged and passed throught FNN to predict the label.     | Cross-attention, then features are either: <br> - concatenated, <br> - or combined via element-wise multiplication.  | BCE |
| **Hierarchical Cross-Attention:** Apply cross-attention to feature maps from different hierarchical levels.      | Hierarchical Cross-Attention, then the feature representations can be either concatenated or upsampled to a common dimensionality, similar to Feature Pyramid Pooling (FPP), to better capture multi-scale dependencies.| BCE |

Among the proposed models, the **Cross-Attention-based architecture** was chosen. Cross-Attention is applied to the feature maps of each image, where each image serves as the attention context for the other. This process generates vector representations, which are then concatenated and passed through Transformer layers and an FNN to predict the binary label. For more ditails view [part2_train_coattention_model.ipynb](part2_train_coattention_model.ipynb)


### 3. Results

**Test dataset** was formed with 20,000 pairs of images containing different shapes and 20,000 pairs with identical images (10,000 per shape).

On the test set, the model achieved:

- Precision: 0.9963084904719146
- Recall: 0.9986
- Accuracy: 0.99745
- F1 Score: 0.9974529291314987
- ROC AUC: 1

The model makes a few minor errors:

- On images with small shapes near the boundary, where transformations cause imprecise object shapes.
- Some errors occur on large/small shapes, which was expected during training due to a strong imbalance in shape sizes. For more ditails view [part1_data_generation.ipynb](part1_data_generation.ipynb)

Improvements:

- Create a balanced dataset for each shape size, which should help eliminate these errors.

---
### Project Layout

**Core Notebooks:**
- **part1_data_generation.ipynb**  
  Visualizes objects with varying parameters (size, position, and rotation angle for squares) to determine constraints for the generation parameters. Also generates training, validation, and test datasets.
- **part2_train_coattention_model.ipynb**  
  Trains the proposed model that incorporates co-attention and self-attention mechanisms.
- **part3_analysis_results.ipynb**  
  Evaluates the model's performance on the test dataset.

**Additional Components:**
- **generate_dataset.py**  
  A script for dataset generation.
- **src/models/**  
  Contains the models used for this task.
- **src/utils/**  
  Contains utility functions and helper scripts.
- **data/**  
  Contains the saved splits for the train, validation, and test datasets.
- **weights/**  
  Contains experiment folders with the saved best weights.
   
---
### To Reproduce The Results:

Create a Python virtual environment and install all required dependencies. We use ipykernel to make the environment available in Jupyter Notebook. Start the jupyter and go through the notebooks.
```bash
python -m venv venv
source venv/bin/activate  
pip install -r requirements.txt
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
jupyter notebook
```
