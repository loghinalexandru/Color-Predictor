using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


class NeuronalNet
{
    public float learningRate = 0.01f;

    public int InputLayerNodes = 2;

    public int HiddenLayerNodes = 2;

    public int OutputLayerNodes = 2;

    private List<List<float>> _layerOneWeights;

    private List<List<float>> _layerTwoWeights;

    private List<float> _layerOneBiases { get; set; }

    private List<float> _layerTwoBiases { get; set; }


    public float Sigmoid(float x)
    {
        return (float)(1 / (1 + Math.Pow(Math.E, (double)-x)));
    }


    public NeuronalNet()
    {

    }

    public NeuronalNet(int inputLayer, int hiddenLayer, int outputLayer)
    {
        this.InputLayerNodes = inputLayer;
        this.HiddenLayerNodes = hiddenLayer;
        this.OutputLayerNodes = outputLayer;
    }

    //Alocate space for the weight's , biases and associate a random number

    public void NetworkInit()
    {
        Random random = new Random();
        this._layerOneWeights = new List<List<float>>(HiddenLayerNodes);
        this._layerTwoWeights = new List<List<float>>(OutputLayerNodes);


        for (int i = 0; i < this.HiddenLayerNodes; ++i)
        {
            List<float> weights = new List<float>(InputLayerNodes);

            for (int j = 0; j < InputLayerNodes; ++j)
            {
                weights.Add((float)random.NextDouble());
            }
            _layerOneWeights.Add(weights);
        }

        for (int i = 0; i < this.OutputLayerNodes; ++i)
        {
            List<float> weights = new List<float>(HiddenLayerNodes);

            for (int j = 0; j < HiddenLayerNodes; ++j)
            {
                weights.Add((float)random.NextDouble());
            }
            _layerTwoWeights.Add(weights);
        }

    }




    public List<float[]> FeedForward(float[] input)
    {
        if (input.Length != InputLayerNodes)
        {
            throw new Exception("Wrong Input/Ouptut Feed !");
        }

        List<float[]> result = new List<float[]>(2); // We have only 2 nets

        //Feeding throught first net
        // First Net
        float[] resultNetOne = new float[HiddenLayerNodes];
        for (int i = 0; i < _layerOneWeights.Count; ++i)
        {
            for (int j = 0; j < input.Length; ++j)
            {
                resultNetOne[i] += input[j] * _layerOneWeights[i][j];
            }
            //resultNetOne[i] += _layerOneBiases[i];
            resultNetOne[i] = Sigmoid(resultNetOne[i]);

        }

        result.Add(resultNetOne);

        //Feeding throught second net
        // Second Net
        float[] resultNetTwo = new float[OutputLayerNodes];
        for (int i = 0; i < _layerTwoWeights.Count; ++i)
        {
            for (int j = 0; j < resultNetOne.Length; ++j)
            {
                resultNetTwo[i] += resultNetOne[j] * _layerTwoWeights[i][j];
            }
            //resultNetTwo[i] += _layerTwoBiases[i];
            resultNetTwo[i] = Sigmoid(resultNetTwo[i]);
        }



        result.Add(resultNetTwo);

        return result;

    }


    //Cost function used = 1/2Sum(targert - output) ^ 2

    private void BackPropagation(float[] firstNetOutput, float[] secondNetOutput, float[] target, float[] input) // Since we have a 3 layer NN the secondNetOutput is the NN output itself
    {
        List<List<float>> secondNetDelta = new List<List<float>>(OutputLayerNodes);
        List<List<float>> firstNetDelta = new List<List<float>>(HiddenLayerNodes);

        //Calculating outputLayer delta weights (secondNetLayer)

        for (int i = 0; i < _layerTwoWeights.Count; ++i)
        {
            List<float> auxiliaryList = new List<float>(HiddenLayerNodes);
            for (int j = 0; j < _layerTwoWeights[i].Count; ++j)
                auxiliaryList.Add((secondNetOutput[i] - target[i]) * secondNetOutput[i] * (1 - secondNetOutput[i]) * firstNetOutput[j]);
            secondNetDelta.Add(auxiliaryList);
        }

        //Calculating hiddenLayer delta (firstNetLayer)

        for (int i = 0; i < _layerOneWeights.Count; ++i)
        {
            List<float> auxiliaryList = new List<float>(InputLayerNodes);
            for (int j = 0; j < _layerOneWeights[i].Count; ++j)
            {
                float previousLayerError = 0;
                for (int z = 0; z < OutputLayerNodes; ++z)
                {
                    // CONTINUA DE AICI NU LUA FORMUAL DE JOS DE BUNA!!!
                    previousLayerError += (secondNetOutput[z] - target[z]) * secondNetOutput[z] * (1 - secondNetOutput[z]) * _layerTwoWeights[z][i];
                }
                auxiliaryList.Add(previousLayerError * firstNetOutput[i] * (1 - firstNetOutput[i]) * input[j]);

            }
            firstNetDelta.Add(auxiliaryList);
        }


        //Adjust weights

        for (int i = 0; i < _layerOneWeights.Count; ++i)
        {
            for (int j = 0; j < _layerOneWeights[i].Count; ++j)
            {
                _layerOneWeights[i][j] = _layerOneWeights[i][j] - firstNetDelta[i][j] * learningRate;
            }
        }


        for (int i = 0; i < _layerTwoWeights.Count; ++i)
        {
            for (int j = 0; j < _layerTwoWeights[i].Count; ++j)
            {
                _layerTwoWeights[i][j] = _layerTwoWeights[i][j] - secondNetDelta[i][j] * learningRate;
            }
        }

    }


    public void TrainNetwork(float[] input, float[] output)
    {

        List<float[]> result = FeedForward(input);
        BackPropagation(result[0], result[1], output, input);

        float error = 0;
        for (int i = 0; i < result[1].Length; ++i)
        {
            error += (result[1][i] - output[i]) * (result[1][i] - output[i]) / 2;
        }

        Console.WriteLine("Error rate : " + error);
    }


}
