
rand = Random.new()

class Layer
  def initialize(topology,layerNum)
    neuronNum = 0
    @neurons = []

    while(neuronNum < topology[layerNum]) do
      numOutputs = layerNum == topology.count - 1 ? 0 : topology[layerNum + 1]
      @neurons.push(Neuron.new(numOutputs,neuronNum,rand))
  #    puts(@@neurons[0].getConnections()[0].getWeight())
      neuronNum = neuronNum + 1

#      puts @@neurons[-1].getWeights()
    end
  end


  def add(neuron)
    @neurons.push(neuron)
  end

  def getNeurons()
    return @neurons
  end
end


class Connection
  def initialize(value)
    @weight = 0.0
    @weight = value
    @deltaweight = 0.0
    @weight = rand()
  end

  def getDW()
    return @deltaweight
  end
  def setDW(val)
    @deltaweight = val
  end

  def getWeight()
    yolo = @weight.to_f()
    return yolo
  end
  def setWeight(value)
    @weight = value
  end
end


class Neuron
  def initialize(numOutputs,myIndex,weight_value)
    @m_myIndex = 0
    @eta = 0.15
    @alpha = 0.5
    @gradient = 0.2
    @m_myIndex = myIndex
    @outputVal = 0.0
    @connections = []

    for i in 0..numOutputs-1 do
      @connections.push(Connection.new(weight_value))
      @connections[-1].setWeight(randomWeight())
    end
  end

  def feedForward(prevLayer)
    sum = 0.0
    for n in 0..prevLayer.getNeurons().count-1 do
#      puts(prevLayer.getNeurons()[n].getOutputVal())
      ting = prevLayer.getNeurons()[n].getConnections()[@m_myIndex]
      this_weight = ting.getWeight()
      sum = sum + (prevLayer.getNeurons()[n].getOutputVal() * this_weight)
    end
    setOutputVal(transferFunction(sum))
  end

  def getOutputVal()
    return @outputVal
  end
  def setOutputVal(n)
    @outputVal = n
  end

  def calcHiddenGradients(nextLayer)
    dow = sumDOW(nextLayer)
    @gradient = dow * transferFunctionDerivative(@outputVal)
  end
  def calcOutputGradients(targetVal)
    @delta = targetVal - @outputVal
    @gradient = @delta* transferFunctionDerivative(@outputVal)
  #  puts(@gradient)
  end

  def updateInputWeights(prevLayer)
    oldDeltaWeight = 0.0
    newDeltaWeight = 0.0
    for n in 0..prevLayer.getNeurons().count-1 do
      neuron = prevLayer.getNeurons()[n]
      oldDeltaWeight = neuron.getConnections[@m_myIndex].getWeight()


      newDeltaWeight = @eta * neuron.getOutputVal() * @gradient + @alpha * oldDeltaWeight


      neuron.getConnections()[@m_myIndex].setDW(newDeltaWeight)
      neuron.getConnections()[@m_myIndex].setWeight(neuron.getConnections()[@m_myIndex].getWeight + newDeltaWeight)
    end
  end

  def randomWeight()
    rand_max = 2147483647.0
    return rand/rand_max
  end

  def sumDOW(nextLayer)
    sum = 0.0
    for n in 0..nextLayer.getNeurons().count-1 do
      sum = sum + m_outputWeights[n].weight * nextLayer.getNeurons()[n].m_gradient
    end
  end

  def transferFunctionDerivative(x)
    return 1.0-x*x
  end

  def transferFunction(x)
    return Math.tanh(x)
  end

  def getConnections()
    return @connections
  end

  def getWeights()
    weights = []
    for i in 0..@connections.count-1 do
      weights.push(@connections[i].getWeight())
    end
    return weights
  end
end

class Network
  def initialize(topology)
    @delta = 0
    @m_layers = []
    @m_recentAverageSmoothingFactor = 100.0
    @m_recentAverageError = 0.0
    @m_error = 0.0

    numLayers = topology.count
    for layerNum in 0..numLayers-1 do
      @m_layers.push(Layer.new(topology,layerNum))
      @m_layers[-1].getNeurons()[-1].setOutputVal(1.0)
    end
  end

  def backPropagate(targetVals)

    outputLayer = @m_layers[-1]
    @m_error = @m_error / outputLayer.getNeurons().count
    @m_error = @m_error ** 0.5
    for n in 0..outputLayer.getNeurons().count-1 do
      @delta = targetVals[n] - outputLayer.getNeurons()[n].getOutputVal()
      @m_error += @delta * @delta
    end
    @m_recentAverageError = (@m_recentAverageError * @m_recentAverageSmoothingFactor + @m_error) / (@m_recentAverageSmoothingFactor + 1.0)
    for n in 0..outputLayer.getNeurons().count-1 do
      outputLayer.getNeurons()[n].calcOutputGradients(targetVals[n])
    end

    layerNum = @m_layers.count-3
    while(layerNum > 0) do
      hiddenLayer = @m_layers[layerNum]
      nextLayer = @m_layers[layerNum + 1]
      for n in 0..hiddenLayer.getNeurons().count-1 do
        hiddenLayer.getNeurons()[n].calcHiddenGradients(nextLayer)
      end
      layerNum = layerNum - 1
    end

    layerNum = @m_layers.count-1

    while(layerNum > 0) do
      layer = @m_layers[layerNum]
      prevLayer = @m_layers[layerNum - 1]
      for n in 0..layer.getNeurons().count-1 do
        layer.getNeurons()[n].updateInputWeights(prevLayer)
      end
      layerNum = layerNum - 1
    end
  end

  def feedForward(inputVals)
    for i in 0..inputVals.count-1 do
      @m_layers[0].getNeurons()[i].setOutputVal(inputVals[i])
    end

    for l in 1..@m_layers.count-1 do
      prevLayer = @m_layers[l-1]
      for n in 0..@m_layers[l].getNeurons().count-1 do

        @m_layers[l].getNeurons()[n].feedForward(prevLayer)
      end
    end
  end

  def getResults(resultVals)
    resultVals.clear
    for n in 0..@m_layers[-1].getNeurons().count-1 do
      resultVals.push(@m_layers[-1].getNeurons()[n].getOutputVal())
    end
  end

  def getLayers()
    all_layers = []
    for i in 0..@m_layers.count-1 do
      all_layers.push(@m_layers[i])
    end
    return all_layers
  end
end

class Computer
  @thisNetwork = 0
  def initialize(topology)
    @thisNetwork = Network.new(topology)
  end

  def BackPropagate(targetVals)
    @thisNetwork.backPropagate(targetVals)
  end

  def getNetwork()
    return @thisNetwork
  end


  def getWeights()
    network = getNetwork()
    layers = network.getLayers()
    weights = []
    for i in 0..layers.count-1 do
      for j in 0..layers[i].getNeurons().count-1 do
        for k in 0..layers[i].getNeurons()[j].getWeights().count-1 do
          weights.push(layers[i].getNeurons()[j].getWeights()[k])
        end
      end
    end
  end


  def feedforward(inputs)
    @thisNetwork.feedForward(inputs)
  end

  def GetResult()
    resultVals = 0.0
    @thisNetwork.getResults(resultVals)
    return resultVals
  end

  def SetWeights(weights)
    @thisNetwork.putWeights(@weights)
  end
end


topology = [3,3,3]
newComputer = Computer.new(topology)
#testingWeights = newComputer.GetWeights()
trainArray = [0.0,1.0,0.0]
testArray = [1.0,1.0,0.0]
#newComputer.SetWeights(testingWeights)
#weights = newComputer.GetWeights()
newComputer.feedforward(trainArray)
c = newComputer.getNetwork().getLayers()[0].getNeurons()[0].getWeights()
#puts(c)
for i in 0..1000 do
  newComputer.BackPropagate(testArray)
  resultVals = []
  resultVals.clear
  newComputer.getNetwork().getResults(resultVals)
  puts(resultVals)
end
