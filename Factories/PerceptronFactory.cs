using System;

namespace NoNuNe {
  
class PerceptronFactory {

  public enum EActivationFunction {
    Sigmoid,
    Tanh,
    ReLU, RectifiedLinearUnit,
    LeakyReLU,
    SoftPlus
  };

  public static NocabRNG rng = new NocabRNG(DateTime.UtcNow); // NocabRNG.defaultRNG; // TODO: Make this more dynamic

  private Func<double, double> activatorFunc = ReLU;
  private Func<Perceptron, double> activatorFuncDerivative = ReLUDerivative;



  public Perceptron buildPerceptron(int layerId = 0, 
                                    int perceptronId = 0,
                                    Func<double, double> activatorFunc = null,
                                    Func<Perceptron, double> activatorFuncDerivative = null) {
      var af = (activatorFunc == null) ? this.activatorFunc : activatorFunc;
      var afd = (activatorFuncDerivative == null) ? this.activatorFuncDerivative : activatorFuncDerivative;
      return new Perceptron(
        initialThreshold: rng.generateDouble(-1, 1),
        activatorFunc: af,
        activatorFuncDerivative: afd
        );
  }

  public void setActivatorFunc(EActivationFunction targetActivationFunc) {
    switch(targetActivationFunc) {
      case EActivationFunction.Sigmoid: {
        this.activatorFunc = sigmoid;
        this.activatorFuncDerivative = sigmoidDerivative;
        break;
      }
      case EActivationFunction.Tanh: {
        this.activatorFunc = tanh;
        this.activatorFuncDerivative = tanhDerivative;
        break;
      }
      case EActivationFunction.RectifiedLinearUnit:
      case EActivationFunction.ReLU: {
        this.activatorFunc = ReLU;
        this.activatorFuncDerivative = ReLUDerivative;
        break;
      }
      case EActivationFunction.LeakyReLU: {
        this.activatorFunc = LeakyReLU;
        this.activatorFuncDerivative = LeakyReLUDerivative;
        break;
      }
      case EActivationFunction.SoftPlus: {
        this.activatorFunc = Softplus;
        this.activatorFuncDerivative = SoftplusDerivative;
        break;
      }
      default:
        // TODO: this case
        break;
    }
  }

  public static float randomWeight() {
    return PerceptronFactory.rng.generateFloat(-1f, 1f, true, true);
  }

  public static double gaussianWeight(float min = -1f, float max = 1f) {
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
