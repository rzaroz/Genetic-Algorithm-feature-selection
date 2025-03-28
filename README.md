# Genetic Algorithm for Feature Selection

This project uses a simple genetic algorithm for feature selection in datasets. The algorithm is used to select the best features for predicting labels using a RandomForest model.

## Installation and Setup

1. First, install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. To run the project, execute the `main.py` file:
   ```bash
   python main.py
   ```

## Description

- **Data:** The data is loaded from a CSV file named `BreastCancerWisconsin(Diagnostic).csv`. This dataset contains various features related to breast cancer.
- **Genetic Algorithm:** The algorithm randomly selects features, combines them, and evaluates prediction accuracy to search for the best feature combination.
- **RandomForest Model:** The RandomForest model is used to evaluate the prediction accuracy.
- **Feature Selection Process:** Features are randomly selected, and in each iteration, new features (offspring) are generated by combining the existing features.

## Code Structure

1. **GeneticEngine Class**:
   - `generate_starter_features`: Generates random starting features.
   - `fitness`: Evaluates the prediction accuracy using the selected features.
   - `mutate`: Alters features to find the optimal combination.
   - `display`: Displays the accuracy of each attempt.
   - `fit`: Runs the genetic algorithm and updates features in each iteration.

## Notes

- This algorithm is designed for learning and experimenting with genetic algorithms and feature selection.
- Make sure the data is correctly loaded, and the CSV file is located in the project directory.
