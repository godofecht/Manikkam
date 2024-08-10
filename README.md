Here's a README file in markdown format for the code:

```markdown
# Simple Neural Network in Ruby

This project implements a basic feedforward neural network with backpropagation in Ruby. The network is structured to learn from provided input data and adjust its weights to minimize the error between its output and the target output.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Classes and Methods](#classes-and-methods)
- [Example](#example)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project consists of the following components:
- A `Layer` class that represents a layer of neurons.
- A `Neuron` class that represents individual neurons, each with connections to the next layer.
- A `Connection` class that handles the weights and delta weights between neurons.
- A `Network` class that ties together the layers to form a complete neural network.
- A `Computer` class that serves as a wrapper to interact with the network, providing methods for training and evaluating the network.

The neural network is trained using a simple form of supervised learning, adjusting its weights to reduce the error between the predicted output and the target output.

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

## Example

Here is an example of how to create and train a network:

```ruby
topology = [3, 3, 3]
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
```

This README provides a comprehensive overview of the project, explains how to use the code, and highlights the key classes and methods. It also includes an example of how to create and train the neural network.
