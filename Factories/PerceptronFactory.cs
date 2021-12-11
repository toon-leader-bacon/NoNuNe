using System;

namespace NoNuNe {
  
public class PerceptronFactory {

  public enum EActivationFunction {
    Sigmoid,
    Tanh,
    ReLU, RectifiedLinearUnit,
    LeakyReLU,
    SoftPlus
  };

  public static NocabRNG rng = NocabRNG.newRNG;

  public static bool gausianWeight = false;

  private Func<double, double> activatorFunc = ReLU;

  private Func<Perceptron, double> activatorFuncDerivative = ReLUDerivative;

  public Perceptron buildPerceptron(int layerId = 0, 
                                    int perceptronId = 0,
                                    Func<double, double> activatorFunc = null,
                                    Func<Perceptron, double> activatorFuncDerivative = null) {
      var af = (activatorFunc == null) ? this.activatorFunc : activatorFunc;
      var afd = (activatorFuncDerivative == null) ? this.activatorFuncDerivative : activatorFuncDerivative;
      return new Perceptron(
        initialThreshold: randomWeight(),
        activatorFunc: af,
        activatorFuncDerivative: afd
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
    this.activatorFunc = getActivatorFunc(targetActivationFunc);
    this.activatorFuncDerivative = getDerivativeFunc(targetActivationFunc);
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

}
}
