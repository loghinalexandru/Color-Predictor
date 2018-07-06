using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Events;

public class ColorGeneration : MonoBehaviour
{

    public Button white;
    public Button black;
    public Button train;
    public Button predict;
    public Text predictText;

    private NeuronalNet nn;

    private List<float[]> input = new List<float[]>();

    private List<float[]> output = new List<float[]>();


    // Use this for initialization


    void Start()
    {

        white.onClick.AddListener(delegate { StoreData(this.white); });
        black.onClick.AddListener(delegate { StoreData(this.black); });
        train.onClick.AddListener(delegate { Train(); });
        predict.onClick.AddListener(delegate { Predict(); });
        white.GetComponent<Image>().color = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value, 1.0f);
        black.GetComponent<Image>().color = white.GetComponent<Image>().color;

    }

    private void Train()
    {
        nn = new NeuronalNet(3, 2, 2);
        nn.NetworkInit();
        nn.learningRate = 0.5f;
        predictText.text = "TRAINING";
        for (int j = 0; j < 10000; ++j)
        {
            for (int i = 0; i < input.Count; ++i)
            {
                nn.TrainNetwork(input[i], output[i]);
            }
        }

        predictText.text = "DONE";


    }

    private void Predict()
    {
        Color color = white.GetComponent<Image>().color;
        List<float[]> result = nn.FeedForward(new float[] { color.r, color.g, color.b });
        if (result[1][0] > result[1][1])
        {
            predictText.text = "White";
        }
        else
        {
            predictText.text = "Black";
        }
    }

    private void StoreData(Button b)
    {
        predictText.text = "Prediction!";
        Color color = b.GetComponent<Image>().color;
        input.Add(new float[] { color.r, color.g, color.b });
        if (b.name == "White")
        {
            output.Add(new float[] { 1.0f, 0f });
        }
        else
        {
            output.Add(new float[] { 0f, 1.0f });
        }
        white.GetComponent<Image>().color = new Color(UnityEngine.Random.value, UnityEngine.Random.value, UnityEngine.Random.value, 1.0f);
        black.GetComponent<Image>().color = white.GetComponent<Image>().color;
    }



}
