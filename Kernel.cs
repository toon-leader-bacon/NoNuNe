

using System.Security.Cryptography;
using System;
using System.Collections.Generic;
namespace NoNuNe {

public class Kernel {

  double[, ,] weights; // 3d array of weights [Width, Height, Depth]

  int width;
  int height;
  int depth;

  /**
   * The Activator Function is used to convert the net value to 
   * the output value.
   * This Func type takes in a the dot product of the input values
   * times the current weights, minus the threshold, also known as
   * the Net value. It returns the Output value of this perceptron.
   * output = activatorFunc(net) 
   *        = activatorFunc([inputs] * [currentWeights] - threshold)
   */
  private Func<double, double> activatorFunc; // TODO: Set this value at construction time

  double threshold;


  public Kernel(double threshold, int width = 3, int height = 3, int depth = 3) {
    this.threshold = threshold;
    this.width = width;
    this.height = height;
    this.depth = depth;

    NocabRNG r = PerceptronFactory.rng;
    weights = new double[width, height, depth];
    for(int w = 0; w < width; w++) {
      for(int h = 0; h < height; h++) {
        for(int d = 0; d < depth; d++) {
          weights[w, h, d] = r.generateDouble(-1, 1);
        }
      }
    }
  }

  public double calculateNet(double[, ,] window) {
    // Assert window has same dimentions as this weights
    double result = 0.0;

    for(int w = 0; w < width; w++) {
      for(int h = 0; h < height; h++) {
        for(int d = 0; d < depth; d++) {
          double weightAtPoint = this.weights[w, h, d];
          double pixelAtPoint = window[w, h, d];
          result += weightAtPoint * pixelAtPoint;
        }
      }
    }

    return result - threshold;
  }

  public double activatonValue(double[, ,] window) {
    double recentNet = this.calculateNet(window);
    return this.activatorFunc(recentNet);
  }

};  // class Kernel


};  // namespace NoNuNe