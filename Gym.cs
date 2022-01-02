using System;
using System.Collections.Generic;

namespace NoNuNe {

public class Gym {
  
  public List<string> classNames = new List<string>();

  public int printEveryN = 999;  // Print confidence logging every 10 training rounds

  public double percentUsedFortraining = 0.90d;

  public Gym() {
    classNames = DataGenerator.indexToName();  // TODO: Make this dynamic
  }

  public void epochTraining(Network network, List<DataPoint> trainingData, int epochs) {
    NocabRNG rng = new NocabRNG(trainingData);
    List<DataPoint> shuffledTrainingData = (List<DataPoint>)rng.shuffleNewList<DataPoint>(trainingData);

    int trainingCount = (int)(shuffledTrainingData.Count * percentUsedFortraining);
    int testingCount = shuffledTrainingData.Count - trainingCount;
    List<DataPoint> _trainingData = shuffledTrainingData.GetRange(0, trainingCount);
    List<DataPoint> _testingData = shuffledTrainingData.GetRange(trainingCount, testingCount);

    for(int i = 0; i < epochs; i++) {
      this.train(network, _trainingData);
    }

    this.test(network, _testingData);
  }

  private void train(Network network, List<DataPoint> trainingData) {
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

  private void test(Network network, List<DataPoint> testingData) {
    PerceptronFactory.ECostFunction cf = PerceptronFactory.ECostFunction.CrossEntropy;

    double maxErrorSoFar = Double.NegativeInfinity;
    double totalError = 0d;

    for(int rep = 0; rep < testingData.Count; rep++) {
      DataPoint dp = testingData[rep]; 
      List<Double> networkOutput = network.evaluate(dp.input);
      List<Double> expectedOutput = dp.expectedOutput;
      double error = PerceptronFactory.calculateError(networkOutput, expectedOutput, cf);

      maxErrorSoFar = Math.Max(maxErrorSoFar, error);
      totalError += error;

    }

    double averageError = totalError / testingData.Count;
    Console.WriteLine($"Maximum Error found: {maxErrorSoFar}");
    Console.WriteLine($"Average Error found: {averageError}");
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
    if (maxIndex < classNames.Count) {
      indexName = classNames[maxIndex];
    }
    Console.WriteLine($"Value '{indexName}' has highest confidence of '{highestConfidence}'");
  }


}; // public class Gym

}; // namespace NoNuNe
