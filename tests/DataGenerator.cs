using System;
using System.Collections.Generic;

namespace NoNuNe {

public static class DataGenerator {

  public static NocabRNG rng = new NocabRNG("DataGenerator");
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
