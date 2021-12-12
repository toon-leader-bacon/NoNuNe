using System.ComponentModel.DataAnnotations.Schema;
using System.Resources;
using System;
using System.Collections.Generic;

namespace NoNuNe {

public static class DataGenerator {

  public static NocabRNG rng = new NocabRNG("DataGenerator");

  public static List<DataPoint> binary0to9(int count) {
    List<DataPoint> result = new List<DataPoint>();

    for(int i = 0; i < count; i++) {
      int value = rng.generateInt(0, 9, true, true);
      result.Add( new DataPoint(input:intToBinary(value), 
                                expectedOutput:intToExpectedOutput(value)));
    }

    return result;
  }

  public static List<string> indexToName() {
    /**
     * The neural net will return a list of confidence values. The index of the
     * highest of these confidence values represents the NN's classification.
     * To convert from an index to a class name, use this list. 
     *
     * For example, if the 3rd element (0 indexed) in the Neural Net's output has a high
     * confidence of 99%, that coresponds to the value "3" in this list. 
     * So the Neural Net things the input value has the class of "3".
     */
    return new List<string>{
      "0",
      "1",
      "2",
      "3",
      "4",
      "5",
      "6",
      "7",
      "8",
      "9"
    };
  }

  public static Tuple<List<double>, List<double>> genInOut() {
    int value = rng.generateInt(0, 9, true, true);
    Console.WriteLine("Generating inOut for value: " + value);
    return new Tuple<List<double>, List<double>>(
      DataGenerator.intToBinary(value), 
      DataGenerator.intToExpectedOutput(value));
  }

  
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

    public static (List<List<double>>, List<int>) digitData(int size) {
      Random rng = new Random();

      List<List<double>> inputData = new List<List<double>>(size);
      List<int> validationData = new List<int>(size);
      for(int i = 0; i < size; i++) {
        int value = rng.Next(0, 10);
        List<double> binaryRep = DataGenerator.intToBinary(value);

        inputData.Add(binaryRep);
        validationData.Add(value);
      }

      return (inputData, validationData);
    }

}; // class DataGenerator

} // namespace NoNuNe
