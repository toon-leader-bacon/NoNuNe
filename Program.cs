using System.Collections.Generic;
using System;


namespace NoNuNe {
  class Program {

    static void PrintListOfDouble(List<double> l) {
      Console.Write("[");
      foreach(double d in l) {
        Console.Write(d);
        Console.Write(", ");
      }
      Console.WriteLine("]");
    }

    static void Main(string[] args) {
      LayerFactory lf = new LayerFactory();
      lf.setActivatorFunc(PerceptronFactory.EActivationFunction.Sigmoid);

      Network n = new Network(0);
      n.appendLayer(lf.buildLayer(layerId: 0, size: 4)); // Input layer
      n.appendLayer(lf.buildLayer(1, 32));
      n.appendLayer(lf.buildLayer(2, 16));
      n.appendLayer(lf.buildLayer(3, 10)); // Output layer
      
      Console.WriteLine("hello Nocab");

      // List<double> input = Program.intToBinary(1);
      // List<double> expectedOutput = Program.intToExpectedOutput(1);

      for(int epoch = 0; epoch < 10000; epoch++) {
        Console.WriteLine();
        var inOuts = DataGenerator.genInOut();
        List<double> output = n.evaluate(inOuts.Item1);

        int maxIndex = -10;
        double maxConfidence = -1d;
        for(int i = 0; i < output.Count; i++) {
          double confidence = output[i];
          
          if(confidence > maxConfidence) {
            maxIndex = i;
            maxConfidence = confidence;
          }
        }
        Console.WriteLine($"{maxIndex}  confidence: {maxConfidence}");

        n.backprop(inOuts.Item2);

      }
      Console.WriteLine("Training done");
    }
  }
}
