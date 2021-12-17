using System;
using System.Collections.Generic;

namespace NoNuNe {
  
public class PerceptronFactory {

  public enum EActivationFunction {
    Sigmoid,
    Tanh,
    ReLU, RectifiedLinearUnit,  // Identical
    LeakyReLU,
    SoftPlus,

    // To represent empty enum value
    NONE
  };

  public enum ECostFunction {
    // Recommended for Regression
    Linear,
    Quadratic, MeanSquaredError,  // Identical

    // Recommended for Classification 
    CrossEntropy,
    HingeLoss,

    // To represent empty enum value
    NONE
  }

  public static NocabRNG rng = NocabRNG.newRNG;

  public static bool gausianWeight = false;

  private EActivationFunction activatorFuncEnum = EActivationFunction.ReLU;
  private ECostFunction costFuncEnum = ECostFunction.Linear;

  public Perceptron buildPerceptron(int layerId = 0, 
                                    int perceptronId = 0,
                                    EActivationFunction activatorFuncEnum = EActivationFunction.NONE,
                                    ECostFunction costFuncEnum = ECostFunction.NONE) {
      EActivationFunction af = (activatorFuncEnum == EActivationFunction.NONE) ? this.activatorFuncEnum: activatorFuncEnum;
      ECostFunction cf = (costFuncEnum == ECostFunction.NONE) ? (this.costFuncEnum) : (costFuncEnum);
      return new Perceptron(
        initialThreshold: randomWeight(),
        activationEnum: af,
        costFunction: cf
      );
  }

  public static Func<double, double> getActivatorFunc(EActivationFunction enumVal) {
    switch(enumVal) {
      case EActivationFunction.Sigmoid: {
        return sigmoid;
      }
      case EActivationFunction.Tanh: {
        return tanh;
      }
      case EActivationFunction.RectifiedLinearUnit:
      case EActivationFunction.ReLU: {
        return ReLU;
      }
      case EActivationFunction.LeakyReLU: {
        return LeakyReLU;
      }
      case EActivationFunction.SoftPlus: {
        return Softplus;
      }
      default:
        // TODO: this case
        // Return default reccomended value :shrug:
        return ReLU;
    }
  }

  public static Func<Perceptron, double> getDerivativeFunc(EActivationFunction enumVal) {
    switch(enumVal) {
      case EActivationFunction.Sigmoid: {
        return sigmoidDerivative;
      }
      case EActivationFunction.Tanh: {
        return tanhDerivative;
      }
      case EActivationFunction.RectifiedLinearUnit:
      case EActivationFunction.ReLU: {
        return ReLUDerivative;
      }
      case EActivationFunction.LeakyReLU: {
        return LeakyReLUDerivative;
      }
      case EActivationFunction.SoftPlus: {
        return SoftplusDerivative;
      }
      default:
        // TODO: this case
        // Return default reccomended value :shrug:
        return ReLUDerivative;
    }
  }

  public void setActivatorFunc(EActivationFunction targetActivationFunc) {
    this.activatorFuncEnum = targetActivationFunc;
  }

  public static Func<double, double, double> getCostFunc(ECostFunction enumVal) {
    switch(enumVal) {
      case ECostFunction.Linear: { return LinearCost; }
      case ECostFunction.Quadratic:
      case ECostFunction.MeanSquaredError: { return QuadraticCost; }

      case ECostFunction.CrossEntropy: { return CrossEntropyCost; }
      case ECostFunction.HingeLoss: { return HingeLossCost; }

      default:
        return CrossEntropyCost;
    }
  }

  public void setCostFunction(ECostFunction targetCostFunction) {
    this.costFuncEnum = targetCostFunction;
  }

  public static double randomWeight() {
    return gausianWeight ? _gaussianWeight() : _randomWeight();
  }

  private static double _randomWeight() {
    return PerceptronFactory.rng.generateDouble(-1f, 1f, true, true);
  }

  private static double _gaussianWeight() {
    double theta = 2 * Math.PI * PerceptronFactory.rng.generateDouble(0, 1);
    double rho = Math.Sqrt(-2 * Math.Log(1 - PerceptronFactory.rng.generateDouble(0, 1)));
    double scale = 0.5f * rho;
    double result = scale * Math.Cos(theta);
    return NocabMathUtility.clamp(result, -1, 1);
  }

#region Perceptron Functions

  public static double sigmoid(double z) {
    /**
     * Near-0 inputs into sigmoid func will lead to near 0.5 output
     * Large positive values => near 1
     * Large negative values => near 0
     */
    return 1d / (1d + Math.Pow(Math.E, -z));
  }

  public static double sigmoidDerivative(Perceptron p) {
    /**
     * NOTE: You MUST provide activationValue = sigmoid(z). In otherwords,
     * provide the output value of the perceptron as the activationValue 
     * variable. 
     */
    return p.recentOutput * (1 - p.recentOutput);
  }

  public static double tanh(double z) {
    /**
     * Near-0 input => 0 output
     * Large positive => 1
     * Large negative => -1
     */
    double powEZ = Math.Pow(Math.E, z);
    double powEnZ = Math.Pow(Math.E, -z);
    return (powEZ - powEnZ) / (powEZ + powEnZ);
  }

  public static double tanhDerivative(Perceptron p) {
    /**
     * NOTE: You MUST provide activationValue = tanh(z). In otherwords,
     * provide the output value of the perceptron as the activationValue 
     * variable. 
     */
    return 1 - (p.recentOutput * p.recentOutput);
  }

  public static double ReLU(double z) {
    return Math.Max(z, 0d);
  }

  public static double ReLUDerivative(Perceptron p) {
    if (p.recentOutput < 0d) { return 0d; }
    return 1d;
  }

  public static double LeakyReLU(double z) {
    if (z < 0) {
      return 0.01 * z; // TODO: make the slope here variable in some way
    }
    return z;
  }

  public static double LeakyReLUDerivative(Perceptron p) {

    if (p.recentOutput < 0d) { return 0.01d; }
    return 1d;
  }

  public static double Softplus(double z) {
    return Math.Log(1d + Math.Pow(Math.E, z));
  }

  public static double SoftplusDerivative(Perceptron p) {
    /**
     * NOTE: You must provide the net value as z. NOT the output value.
     */
    return 1d / (1 + Math.Pow(Math.E, -p.RecentNet));
  }

#endregion Perceptron Functions

#region Cost Calculation Functions

  private static double LinearCost(double estimate, double label) {
    return label - estimate;
  }

  private static double QuadraticCost(double estimate, double label) {
    // Mean Squared Error for a single output layer perceptron
    return (label - estimate) * (label - estimate);
  }

  private static double CrossEntropyCost(double estimate, double label) {
    return -1d * (
      (label * Math.Log(estimate)) + 
      ((1d - label) * Math.Log(1d - estimate)));
  }

  private static double HingeLossCost(double estimate, double label) {
    return Math.Max(0, 1 - (label * estimate));
  }

  private static double QuadraticCost(List<Double> estimate, List<Double> label) {
    /**
     * Mean Squared error
     * return (1/n)*Sum((estimate[i] - label[i])^2)
     * 
     * TODO: Not sure what to do if the two arrays are different sized...
     */
    double result = 0d;

    int minCount = Math.Min(estimate.Count, label.Count);
    for(int i = 0; i < minCount; i++ ) {
      double estimateI = estimate[i];
      double labelI = label[i];

      result += (labelI - estimateI) * (labelI - estimateI);
    }

    return result / minCount;
  }

  private static double CrossEntropyCost(List<Double> estimate, List<Double> label) {
    /**
     * TODO: Not sure what to do if the two arrays are different sized...
     */
    double result = 0d;
    int minCount = Math.Min(estimate.Count, label.Count);
    for(int i = 0; i < minCount; i++) {
      double estimateI = estimate[i];
      double labelI = label[i];

      result += (labelI * Math.Log(estimateI)) + 
                ((1 - labelI) * Math.Log(1 - estimateI));
    }

    return result / -minCount;
  }

#endregion Cost Calculation Functions 

}
}
