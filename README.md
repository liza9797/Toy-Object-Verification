# Toy-Object-Verification

**Objective:**  
Train a classifier designed to take two images as input and predict whether the shapes depicted in them are identical.


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
   
