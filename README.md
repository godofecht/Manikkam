# Simple Neural Network in Ruby

This project aims to provide robust support for perceptron and feedforward neural networks (FNNs) in Ruby. It starts with the implementation of the perceptron, a basic building block of neural networks, and extends to more complex feedforward neural network architectures. This project is part of an experiment to implement these concepts across multiple programming languages, offering a practical comparison of their implementation in Ruby. Future extensions will build on this foundation to explore more advanced neural network architectures like Recurrent Neural Networks (RNNs) and beyond.

## Table of Contents

- [Overview](#overview)
- [Feedforward Neural Networks (FNN)](#feedforward-neural-networks-fnn)
- [Backpropagation](#backpropagation)
- [Recurrent Neural Networks (RNN)](#recurrent-neural-networks-rnn)
- [Installation](#installation)
- [Usage](#usage)
- [Classes and Methods](#classes-and-methods)
- [Checklist](#checklist)
- [Example](#example)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is an introduction to neural networks in Ruby, focusing on implementing perceptrons and feedforward neural networks. It explores the foundational concepts of neural networks, such as the perceptron model, and builds towards more complex architectures. This project is part of a broader experiment to understand and compare the implementation of neural networks in different programming languages.

## Feedforward Neural Networks (FNN)

Feedforward Neural Networks (FNNs) are the simplest type of artificial neural network. In an FNN, information moves in one direction—from the input layer, through the hidden layers, and finally to the output layer. There are no cycles or loops in the network. Each layer consists of neurons that apply a weighted sum of their inputs followed by an activation function to produce an output.

### Key Concepts:
- **Input Layer**: The initial layer where data is fed into the network.
- **Hidden Layers**: Intermediate layers that transform the input into a form the output layer can use.
- **Output Layer**: The final layer that produces the output of the network.
- **Activation Function**: A function applied to each neuron's output to introduce non-linearity into the network, enabling it to learn more complex patterns. Common activation functions include Sigmoid, Tanh, and ReLU.

## Backpropagation

Backpropagation is the algorithm used to train feedforward neural networks. It works by calculating the gradient of the loss function with respect to each weight by the chain rule, iterating backward from the output layer to the input layer. This process allows the network to adjust its weights to minimize the error between the predicted output and the actual target.

### Key Concepts:
- **Loss Function**: A function that measures how well the network's predictions match the target values. Common loss functions include Mean Squared Error (MSE) and Cross-Entropy.
- **Gradient Descent**: An optimization algorithm used to minimize the loss function by adjusting the weights in the network. Variants include Stochastic Gradient Descent (SGD), Momentum, and Adam.
- **Learning Rate**: A hyperparameter that controls the step size during the gradient descent update.

## Recurrent Neural Networks (RNN)

Recurrent Neural Networks (RNNs) are a class of neural networks where connections between neurons form a directed cycle. This architecture allows the network to maintain a memory of previous inputs, making it suitable for tasks involving sequential data, such as time series prediction, natural language processing, and speech recognition.

### Key Concepts:
- **Recurrent Connections**: Unlike FNNs, RNNs have connections that loop back on themselves, enabling the network to maintain a state or memory of previous inputs.
- **Hidden State**: A vector that captures the information from previous time steps in the sequence.
- **Vanishing and Exploding Gradients**: Challenges associated with training RNNs using backpropagation through time (BPTT), where gradients can become very small (vanishing) or very large (exploding), making training difficult.

### Future Extensions:
- **Long Short-Term Memory (LSTM)**: A type of RNN designed to overcome the vanishing gradient problem, capable of learning long-term dependencies.
- **Gated Recurrent Unit (GRU)**: A simpler alternative to LSTM with similar performance, using fewer parameters.

## Installation

To run this code, you need to have Ruby installed on your system. You can download Ruby from the [official website](https://www.ruby-lang.org/en/downloads/).

## Usage

1. Clone or download this repository.
2. Navigate to the directory containing the code.
3. Modify the `topology`, `trainArray`, and `testArray` variables in the script as needed.
4. Run the script using Ruby:

   ```bash
   ruby your_script_name.rb
   ```

Replace `your_script_name.rb` with the name of the file containing the code.

## Classes and Methods

### `Layer`
Represents a single layer in the neural network, consisting of multiple neurons.

- **`initialize(topology, layerNum)`**: Creates a layer with the specified topology and layer number.
- **`add(neuron)`**: Adds a neuron to the layer.
- **`getNeurons()`**: Returns the neurons in the layer.

### `Neuron`
Represents a single neuron, including its connections to the next layer.

- **`initialize(numOutputs, myIndex, weight_value)`**: Initializes the neuron with the specified number of outputs, its index, and initial weight value.
- **`feedForward(prevLayer)`**: Feeds forward the input from the previous layer, calculating the output value.
- **`calcHiddenGradients(nextLayer)`**: Calculates the gradients for a hidden neuron based on the next layer.
- **`calcOutputGradients(targetVal)`**: Calculates the gradient for an output neuron based on the target value.
- **`updateInputWeights(prevLayer)`**: Updates the input weights from the previous layer based on the calculated gradients.
- **`getOutputVal()`**: Returns the output value of the neuron.
- **`setOutputVal(n)`**: Sets the output value of the neuron.
- **`getConnections()`**: Returns the connections from this neuron to the next layer.
- **`getWeights()`**: Returns the weights of the connections.
- **`randomWeight()`**: Generates a random weight for a connection.
- **`transferFunction(x)`**: Applies the transfer function (tanh) to the input `x`.
- **`transferFunctionDerivative(x)`**: Applies the derivative of the transfer function to `x`.

### `Connection`
Represents the connection between two neurons, holding the weight and delta weight.

- **`initialize(value)`**: Initializes the connection with a weight value.
- **`getDW()`**: Returns the delta weight.
- **`setDW(val)`**: Sets the delta weight.
- **`getWeight()`**: Returns the weight.
- **`setWeight(value)`**: Sets the weight.

### `Network`
Represents the entire neural network, consisting of multiple layers.

- **`initialize(topology)`**: Initializes the network with the specified topology.
- **`feedForward(inputVals)`**: Feeds input values through the network to produce an output.
- **`backPropagate(targetVals)`**: Adjusts the weights of the network to minimize the error between the output and target values.
- **`getResults(resultVals)`**: Gets the output results from the network.
- **`getLayers()`**: Returns all the layers of the network.

### `Computer`
A wrapper class to interact with the network.

- **`initialize(topology)`**: Initializes the computer with a neural network of the specified topology.
- **`BackPropagate(targetVals)`**: Trains the network with the target values using backpropagation.
- **`feedforward(inputs)`**: Feeds input values through the network.
- **`GetResult()`**: Retrieves the output results from the network.
- **`getNetwork()`**: Returns the network object.
- **`getWeights()`**: Returns the weights of the network's connections.
- **`SetWeights(weights)`**: Sets the weights of the network's connections (incomplete).

## Checklist

### Completed Features
- [x] Implemented `Layer` class with methods to manage neurons.
- [x] Implemented `Neuron` class with feedforward and backpropagation methods.
- [x] Implemented `Connection` class to handle weights and delta weights between neurons.
- [x] Implemented `Network` class to manage layers and propagate values.
- [x] Implemented `Computer` class to interact with the neural network.
- [x] Created a basic example to demonstrate network creation, training, and result retrieval.
- [x] Explained Feedforward Neural Networks (FNN).
- [x] Explained Backpropagation algorithm.

### Todos
- [ ] Complete the `SetWeights` method in the `Computer` class.
- [ ] Address potential issues with the `GetResult` method in the `Computer` class.
- [ ] Implement or correct the `sumDOW` method to support gradient calculations.
- [ ] Add additional tests and validation for edge cases.
- [ ] Optimize performance for larger networks and datasets.
- [ ] Improve documentation for complex methods and concepts.
- [ ] Explore Recurrent Neural Networks (RNN) implementation.
- [ ] Add LSTM and GRU implementations for handling sequential data.

## Example

Here is an example of how to create and train a network:

```ruby
topology = [3, 3

, 3]
newComputer = Computer.new(topology)
trainArray = [0.0, 1.0, 0.0]
testArray = [1.0, 1.0, 0.0]

for i in 0..100 do
  newComputer.feedforward(trainArray)
  newComputer.BackPropagate(testArray)
  resultVals = []
  newComputer.getNetwork().getResults(resultVals)
  puts(resultVals)
end
```

This example creates a network with 3 layers, each containing 3 neurons. It then trains the network on the `trainArray` input, adjusting the weights to minimize the error compared to `testArray`.

## Notes

- The code has some issues that need to be addressed, such as the incomplete `sumDOW` method and potential errors in the `GetResult` and `SetWeights` methods.
- The network uses the `tanh` function as the transfer function, and its derivative for backpropagation.
- The random weight generation uses a simple normalization approach.

## Contributing

If you would like to contribute to this project, please fork the repository, make your changes, and submit a pull request.

## License

This project is open-source and available under the MIT License.
