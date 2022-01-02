using System.Threading.Tasks.Dataflow;
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
      lf.setCostFunction(PerceptronFactory.ECostFunction.HingeLoss);

      Network n = new Network(0);
      n.appendLayer(lf.buildLayer(layerId: 0, size: 4, perceptronActivation: PerceptronFactory.EActivationFunction.ReLU)); // Input layer
      n.appendLayer(lf.buildLayer(1, 32, PerceptronFactory.EActivationFunction.LeakyReLU));
      // n.appendLayer(lf.buildLayer(2, 16, PerceptronFactory.EActivationFunction.ReLU));
      n.appendLayer(lf.buildLayer(2, 10, PerceptronFactory.EActivationFunction.Sigmoid)); // Output layer
      
      Console.WriteLine("hello Nocab");

      List<DataPoint> data = DataGenerator.binary0to9(50000);
      Gym gym = new Gym();
      gym.epochTraining(n, data, 15);
      Console.WriteLine("Training done");
    }
  }
}
