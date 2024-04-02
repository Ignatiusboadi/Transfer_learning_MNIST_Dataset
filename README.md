## Overview
This project focuses on implementing transfer learning techniques on the MNIST dataset. It comprises three main components:

1. **Building softmax regression models**: Two separate softmax regression models are constructed to predict even numbers and odd numbers, utilizing the mnist even and mnist odd datasets, respectively.

2. **Creating a neural network with one hidden Layer**: A neural network with one hidden layer consisting of 500 neurons is implemented to perform a similar task as the softmax regression models in the first part.

3. **Transfer Learning**: Transfer learning is implemented by training a softmax regression model using features extracted from the neural network trained in the second part. Specifically, features obtained from the odd dataset are utilized to train a model for predicting even numbers, and features from the even dataset are used to predict odd numbers.

## Project Structure
The project consists of four main files:

1. `NeuralNetwork.py`: Contains the neural network class.
2. `SoftmaxRegression.py`: Contains the softmax regression class.
3. `dataset.py`: Prepares and processes the datasets.
4. `main.py`: Instantiates all the classes and runs the project.

## Dependencies
To run this project, the following dependencies are required:

1. PyTorch
2. Matplotlib
3. NumPy
4. Pandas
5. scikit-learn (sklearn)

## Setup and Installation
Follow these steps to set up and run the project:

1. Clone the repository:
   ```
   git clone https://github.com/Ignatiusboadi/Transfer_learning_MNIST_Dataset.git
   ```

2. Navigate to the project directory:
   ```
   cd Transfer_learning_MNIST_Dataset
   ```

3. Install dependencies:
   ```
   pip install torch torchvision matplotlib numpy pandas scikit-learn
   ```

4. Run the project:
   ```
   python main.py
   ```

## Usage
Upon running `main.py`, the project will execute the three parts described above. Make sure to review the code for any specific configurations or parameters you may want to adjust.

## License
This project is licensed under the [MIT License](LICENSE).
