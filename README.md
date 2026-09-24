# 🏠 Real Estate Price Prediction: NumPy Neural Network From Scratch

## About This Project
This project was developed to demonstrate a deep mathematical understanding of neural network mechanics. Rather than relying on black-box deep learning frameworks like PyTorch or TensorFlow, this regression model is built entirely from scratch using Python and **NumPy matrix operations**. 

It predicts continuous real estate prices based on 6 input features using a custom **6 → 8 → 1 architecture**. By manually implementing forward propagation, mean squared error loss, backpropagation (calculus chain rule), and stochastic gradient descent, the model successfully overcomes the vanishing gradient problem using **Leaky ReLU** and **He Initialization**.

## Architecture Overview
![Custom Neural Network Forward Structure](<nn output.png>)

The custom neural network forward structure is explicitly designed for continuous regression[cite: 1]:
* **Input Layer (6 Neurons):** Ingests scaled data features, represented as nodes X1 through X6[cite: 1].
* **Hidden Layer (8 Neurons):** Extracts complex, non-linear patterns using nodes H1 through H8 and a **Leaky ReLU** activation function to prevent "dying" neurons during training[cite: 1].
* **Output Layer (1 Neuron):** Aggregates hidden signals into a single continuous price prediction, represented by node Y, using a **Linear** activation function[cite: 1].

## Technical Highlights
* **Manual Backpropagation:** Gradients for weights and biases are calculated mathematically using NumPy dot products (`np.dot`) and transposed matrices, passing the error backward layer by layer.
* **He Initialization:** Weight matrices ($W_1$, $W_2$) are initialized using the mathematically optimized He formula (`np.sqrt(2 / input_size)`) to ensure the network begins learning rapidly from Epoch 1 without exploding gradients.
* **Strict Data Leakage Prevention:** `StandardScaler` is strictly fitted (`fit_transform`) to the training data and only applied (`transform`) to the testing data.
* **Stochastic Shuffling:** The training loop utilizes row permutation to randomize data feeding, preventing the model from memorizing the sequence of the dataset.

## Final Model Evaluation
The network was evaluated on a 20% holdout test set (83 unseen houses) to ensure real-world generalization.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **R² (Variance Explained)** | **78.5%** | The 6 input features successfully explain nearly 80% of the underlying real estate market trends. |
| **15% Tolerance Accuracy** | **73.5%** | Nearly 3 out of every 4 predictions fall within a strict 15% margin of the true real estate value. |
| **MAE (Mean Absolute Error)** | **± 4.35** | On average, the model's price prediction deviates by exactly 4.35 units. |
| **RMSE (Root Mean Squared)** | **± 6.01** | Heavily penalizes large prediction errors; confirms major outliers are kept to a minimum. |

## Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/Aiman-Al-Mahmud/real-estate-neural-network-numpy.git
cd real-estate-neural-network-numpy

**2. Install dependencies**
```bash
pip install -r requirements.txt
Bash

**3. Run the model**
Open real_estate_nn.ipynb in Jupyter Notebook, Google Colab, or VS Code and execute the cells to observe the manual training loop and real-time loss reduction.
