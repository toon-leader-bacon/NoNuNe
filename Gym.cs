using System.Xml;
using System.Data;
using System;
using System.Collections.Generic;

namespace NoNuNe {

public class Gym {
  
  public List<string> outputLookup = new List<string>();

  public int printEveryN = 100;  // Print confidence logging every 10 training rounds

  public Gym() {
    outputLookup = new List<string>{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
  }

  public void epochTraining(Network network, List<DataPoint> trainingData, int epochs) {
    for(int i = 0; i < epochs; i++) {
      this.train(network, trainingData);
    }
  }

  public void train(Network network, List<DataPoint> trainingData) {
    for (int rep = 0; rep < trainingData.Count; rep++) {
      DataPoint dp = trainingData[rep];
      List<Double> actualOutput = network.evaluate(dp.input);

      if(rep % printEveryN == 0) {
        PrintHighestConfidence(actualOutput);
        PrintHighestConfidence(dp.expectedOutput);
        Console.WriteLine("\n\n");
      }

      network.backprop(dp.expectedOutput);
    }
  }

  public void PrintHighestConfidence(List<double> confidenceOutput) {
    int maxIndex = -1;
    double highestConfidence = -10000d;
    for(int i = 0; i < confidenceOutput.Count; i++) {
      double confidence = confidenceOutput[i];
      if (confidence > highestConfidence) {
        maxIndex = i;
        highestConfidence = confidence;
      }
    }
    
    string indexName = $"Invalid index {maxIndex}";
    if (maxIndex < outputLookup.Count) {
      indexName = outputLookup[maxIndex];
    }
    Console.WriteLine($"Value '{indexName}' has highest confidence of '{highestConfidence}'");
  }


}; // public class Gym

}; // namespace NoNuNe
