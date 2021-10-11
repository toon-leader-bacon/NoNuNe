using System.Collections.Generic;
using System;


namespace NoNuNe
{
  class Program
  {

    public static List<double> intToBinary(int value) {
      // Smallest digit is the furthest right
      // The value 8 => [1, 0, 0, 0]
      // The value 5 => [0, 1, 0, 1]
      List<double> result = new List<double>{0, 0, 0, 0};

      int curIndex = 3;
      while(curIndex >= 0) {

        result[curIndex] = (value % 2 == 0) ? 0d : 1d;
        value = value / 2;
        curIndex--;
      }

      return result;
    }

    public static List<double> intToExpectedOutput(int value) {
      switch(value) {
        case 0: 
          return new List<double>{1d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d};
        case 1: 
          return new List<double>{0d, 1d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d};
        case 2: 
          return new List<double>{0d, 0d, 1d, 0d, 0d, 0d, 0d, 0d, 0d, 0d};
        case 3: 
          return new List<double>{0d, 0d, 0d, 1d, 0d, 0d, 0d, 0d, 0d, 0d};
        case 4: 
          return new List<double>{0d, 0d, 0d, 0d, 1d, 0d, 0d, 0d, 0d, 0d};
        case 5: 
          return new List<double>{0d, 0d, 0d, 0d, 0d, 1d, 0d, 0d, 0d, 0d};
        case 6: 
          return new List<double>{0d, 0d, 0d, 0d, 0d, 0d, 1d, 0d, 0d, 0d};
        case 7: 
          return new List<double>{0d, 0d, 0d, 0d, 0d, 0d, 0d, 1d, 0d, 0d};
        case 8: 
          return new List<double>{0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 1d, 0d};
        case 9:
        default: 
          return new List<double>{0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 0d, 1d};
      };
    }

    static (List<List<double>>, List<int>) digitData(int size) {
      Random rng = new Random();

      List<List<double>> inputData = new List<List<double>>(size);
      List<int> validationData = new List<int>(size);
      for(int i = 0; i < size; i++) {
        int value = rng.Next(0, 10);
        List<double> binaryRep = Program.intToBinary(value);

        inputData.Add(binaryRep);
        validationData.Add(value);
      }

      return (inputData, validationData);
    }

    static void PrintListOfDobule(List<double> l) {
      Console.Write("[");
      foreach(double d in l) {
        Console.Write(d);
        Console.Write(", ");
      }
      Console.WriteLine("]");
    }

    static void Main(string[] args) {
      Layer l0 = new Layer(0, 4);  // Input layer
      Layer l1 = new Layer(1, 32);
      //Layer l2 = new Layer(2, 8);
      Layer l3 = new Layer(2, 10); // Output layer

      Network n = new Network(0);
      n.appendLayer(l0);
      n.appendLayer(l1);
      //n.appendLayer(l2);
      n.appendLayer(l3);
      
      Console.WriteLine("hello Nocab");
      Console.WriteLine("Evaluating the first input:");

      List<double> input = Program.intToBinary(1);
      List<double> expectedOutput = Program.intToExpectedOutput(1);

      int maxIndex = -10;
      double maxConfidence = -1d;
      for(int epoch = 0; epoch < 1000; epoch++) {
        List<double> output = n.evaluate(input);

        for(int i = 0; i < output.Count; i++) {
          double confidence = output[i];
          if(confidence > maxConfidence) {
            maxIndex = i;
            maxConfidence = confidence;
          }
        }

        
        n.backprop(expectedOutput);

      }
      
      Console.WriteLine($"Index = '{maxIndex}' with a confidence of '{maxConfidence}'");
      Console.WriteLine("Training done");
    }
  }
}
