
class Layer
  def initialize()
    @@neurons = []
  end


  def add(neuron)
    @@neurons.push(neuron)
  end

  def getNeurons()
    return @@neurons
  end
end


class Connection
  @@weight = 0
  @@deltaweight = 0
end

class Neuron
  @@eta = 0
  @@alpha = 0

  @@m_gradient = 0
  @@m_outputVal = 0
  @@m_myIndex = 0





  def initialize(numOutputs,myIndex)
    @@m_outputWeights = []
    for i in 0..numOutputs-1 do
      @@m_outputWeights.push(Connection.new())
      @@m_outputWeights[-1] = randomWeight()
    end
    @@m_myIndex = myIndex
  end

  def feedForward(prevLayer)
    sum = 0.0

    for n in 0..prevLayer.count-1 do
      sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_myIndex].weight
    end

    @@m_outputVal = transferFunction(sum)
  end

  def getOutputVal()
    return @@m_outputVal
  end

  def setOutputVal(n)
    @@m_outputVal = n
  end

  def calcHiddenGradients(nextLayer)
    dow = sumDOW(nextLayer)
    m_gradient = dow * transferFunctionDerivative(m_outputVal)
  end

  def calcOutputGradients(targetVal)
    delta = targetVal - m_outputVal
    m_gradient = delta* transferFunctionDerivative(m_outputVal)
  end

  def updateInputWeights(prevLayer)
    for n in 0..prevLayer.count-1 do
      neuron = prevLayer[n]
      oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaweight

      double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight

      neuron.m_outputWeights[m_myIndex].deltaweight = newDeltaWeight
      neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight
    end
  end

  def randomWeight()
    rand_max = 2147483647
    return (rand()/rand_max)
  end

  def sumDOW()
    sum = 0.0
    for n in 0..nextLayer.count-1 do
      sum = sum + m_outputWeights[n].weight * nextLayer[n].m_gradient
    end
  end

  def transferFunctionDerivative(x)
    return 1-x*x
  end

  def transferFunction(x)
    return tanh(x)
  end

  def getWeights()
    return @@m_outputWeights
  end
end

class Network

  @@layer = []

  @@m_layers = []
  @@m_layers.push(Layer.new())

  m_recentAverageSmoothingFactor = 100.0
  m_recentAverageError = 0
  m_error = 0
  m_gradient = 0

  def initialize(topology)
    numLayers = topology.count
    layerNum = 0
    while layerNum < numLayers do
      @@m_layers.push(Layer.new())
      numOutputs = layerNum == topology.count - 1 ? 0 : topology[layerNum + 1]

      neuronNum = 0
      while(neuronNum <= topology[layerNum]) do
        @@m_layers[-1].add(Neuron.new(numOutputs,neuronNum))
        neuronNum = neuronNum + 1
      end

     @@m_layers[-1].getNeurons()[-1].setOutputVal(1.0)


     layerNum = layerNum + 1

    end
  end

  def backPropagate(targetVals)
    outputLayer = m_layers[-1]
    error = 0

    for n in 0..outputLayer.count-1 do
      delta = targetVals[n] - outputLayer[n].getOutputVal()
      error += delta * delta
    end

    error = error / outputLayer.count-1
    error = sqrt(error)


    m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + error) / (m_recentAverageSmoothingFactor + 1.0)


    for n in 0..outputLayer.count-1 do
      outputLayer[n].calcOutputGradients(targetVals[n])
    end

    while(layerNum > 0) do
      hiddenLayer = m_layers[layerNum]
      nextlayer = m_layers[layerNum + 1]

      for n in 0..hiddenlayer.count-1 do
        hiddenLayer[n].calcHiddenGradients(nextLayer)
      end

      layerNum = layerNum - 1
    end

  while(layerNum > 0) do
    layer = m_layers[layerNum]
    prevLayer = m_layers[layerNum - 1]

    for n in 0..layer.count-1 do
      layer[n].updateInputWeights[prevLayer]
    end
  end


  end

  def feedForward(inputVals)
  end

  def getResults(resultVals)
    resultVals.clear
    for n in 0..@@m_layers[-1].getNeurons().count-1 do
      resultVals.push(@@m_layers[-1].getNeurons()[n].getOutputVal())
    end
  end

  def getWeights()
    weights = []
    for i in 0..@@m_layers.count-1 do
      numNeurons = @@m_layers[i].getNeurons().count
      puts(@@m_layers[i].getNeurons()[i].getWeights())
      for j in 0..numNeurons-1 do
        numWeights = @@m_layers[i].getNeurons().count
        for k in 0..numWeights-1 do
          weights.push(@@m_layers[i].getNeurons()[j].getWeights().count)
        end
      end
    end
    return weights
  end

  def putWeights(weights)
    cWeight = 0
    for i in 0..@@m_layers.count-1 do
      numNeurons = @@m_layers[i].getNeurons().count
      puts(@@m_layers[i].getNeurons()[i].getWeights())
      for j in 0..numNeurons-1 do
        numWeights = @@m_layers[i].getNeurons().count
        for k in 0..numWeights-1 do
          cWeight = cWeight + 1
          @@m_layers[i].getNeurons()[j].getWeights()[k] = weights[cWeight]
        end
      end
    end
    return
  end
end

class Computer

  @@fitness = 0
  @@thisNetwork = 0
  @@topology = []
  @@weights = []


  def initialize(topology)
    setNetwork(topology)
  end

  def setNetwork(top)
    @@thisNetwork = Network.new(top)
  end

  def BackPropagate(targetVals)
    thisNetwork.backPropagate(targetVals)
  end

  def getNetwork()
    return @@thisNetwork
  end

  def GetFitness()
    return @@fitness
  end

  def GetWeights()
    weights = @@thisNetwork.getWeights()
    return weights
  end

  def feedforward(inputs)
    @@thisNetwork.feedForward(inputs)
  end

  def GetResult()
    resultVals = 0
    thisNetwork.getResults(resultVals)
    return resultVals
  end

  def SetFitness()
    @@fitness = n
  end

  def SetWeights(weights)
    @@thisNetwork.putWeights(weights)
  end
end


topology = [4,4,4]
newComputer = Computer.new(topology)
testingWeights = newComputer.GetWeights()
testArray = [1,1,0,1]
newComputer.SetWeights(testingWeights)
newComputer.feedforward(testArray)
resultVals = []
resultVals.clear
newComputer.getNetwork().getResults(resultVals)
puts(resultVals)
