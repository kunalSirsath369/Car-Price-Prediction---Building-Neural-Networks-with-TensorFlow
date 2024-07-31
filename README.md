# Car Price Prediction using TensorFlow

This project aims to build a neural network model using TensorFlow to predict car prices based on various features such as years, kilometers driven, rating, condition, economy, top speed, horsepower, and torque.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

The goal of this project is to develop a machine learning model that can accurately predict the price of a car given certain features. We use a neural network model built with TensorFlow for this task.

## Dataset

The dataset used in this project contains various features of cars and their corresponding prices. The main features include:
- Years
- Kilometers Driven
- Rating
- Condition
- Economy
- Top Speed
- Horsepower
- Torque
- Current Price

## Installation

To run this project, you need to have Python and TensorFlow installed. Follow these steps to set up the environment:

1. Clone the repository:
   \`\`\`sh
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction
   \`\`\`

2. Create a virtual environment and activate it:
   \`\`\`sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   \`\`\`

3. Install the required packages:
   \`\`\`sh
   pip install -r requirements.txt
   \`\`\`

## Usage

1. Ensure you have the dataset file \`train.csv\` in the project directory.

2. Run the \`car_price_prediction.py\` script to train and evaluate the model:
   \`\`\`sh
   python car_price_prediction.py
   \`\`\`

3. The training process will output the loss and accuracy metrics and save the trained model.

## Model Architecture

The neural network model is built using TensorFlow and consists of the following layers:
- Input Layer
- Normalization Layer
- Three Dense Layers with ReLU activation
- Output Dense Layer

## Training

The model is trained using the following configurations:
- Loss Function: Mean Absolute Error
- Optimizer: Adam
- Metrics: Root Mean Squared Error
- Epochs: 100
- Batch Size: 32

## Evaluation

The model is evaluated using the validation dataset, and the performance is monitored using loss and Root Mean Squared Error (RMSE) metrics.

## Results

The training and validation loss and RMSE are plotted to visualize the model's performance over epochs.
