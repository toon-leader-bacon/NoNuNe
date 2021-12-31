
using System;
using System.Collections.Generic;

namespace NoNuNe {

public class Optimizer {

  double learningRate = 0.01d; 

  int maxMomentumCount = 5;

  WeightMatrix wm;

  List<double> perceptronsNewWeights = new List<double>(); 


  private void blab(Perceptron p, Layer leftLayer, double expectedOutput) {
    /**
     * Output layer update weights function. 
     * See _updateWeights(...) for the math.
     */
    double lowercaseDelta = output_dETot_over_dOut(p, expectedOutput) * dOut_over_dNet(p);
    List<double> newWeights = prepareNewWeights(p, leftLayer, lowercaseDelta);
  }

  private List<double> prepareNewWeights(Perceptron p, Layer leftLayer, double lowercaseDelta) {
    /**
     * Chain rule: 
     *  dETotal    dETotal   dOutput      dNet                         dNet
     * --------- = ------- * ------- * --------- = lowercaseDelta * ---------
     * dWeight_i   dOutput    dNet     dWeight_i                    dWeight_i
     *
     * New Weight_i = old_weight_i - (learningRate * (dETotal/ dWeight_i))
     * 
     * Notice:
     *  dETotal   dOutput
     *  ------- * ------- = lowercaseDelta => The same value for every input weight_i
     *  dOutput    dNet 
     */
    List<double> result = new List<double>();
    for(int weightIndex = 0; weightIndex < p._currentWeights.Count; weightIndex++) {
      double momentum = simpleAverage(wm.getRecentForWeight(weightIndex, 5));

      // Calculate the weight change for the current location      
      double dETot_over_dWeight = lowercaseDelta * dNet_over_dWeight(leftLayer, weightIndex);
      double weightChange = this.learningRate * dETot_over_dWeight;
      wm.appendWeight(weightIndex, weightChange);

      double newWeight = p._currentWeights[weightIndex] - (weightChange + momentum);
      result.Add(newWeight);
    }
    return result;
  }

  double simpleAverage(List<double> weights) {
    double sum = 0d;
    foreach(double d in weights) { sum += d; }
    return sum / weights.Count;
  }

#region Back Propagation Math

  private double dNet_over_dWeight(Layer leftLayer, int indexOfWeight) {
    /**
     *   dNet
     * -------- = output of left Perceptron.
     * dWeight_i
     *
     * (leftPerceptron_0) --Weight_0----V 
     * (leftPerceptron_1) --Weight_1--> (thisPerceptron)
     * (leftPerceptron_i) --Weight_i----^ 
     *
     *
     * (0.5) --0.2----V 
     * (0.1) --0.4--> (thisPerceptron)
     * (0.8) --0.3----^
     *
     * Given the above left layer and index of 0, this function will return 0.5 because
     * the output value of the left layer's 0th perceptron is 0.5.
     * Similarly, for index i, the i-th perceptron's output will be returned. In the
     * above example that would be 0.8.
     */
    // Result = recent output of the perceptron for the target weight
    return leftLayer.getPerceptron(indexOfWeight).recentOutput;
  }

  

  private double output_dETot_over_dOut(Perceptron p, double expectedOutput) {
    /**
     * dErrorTotal
     * ----------- = -(expected_output - actual_output) = actual_output - expected_output
     *   dOutput
     * 
     * "How much does the total error change with respect to the output?"
     *
     * How far off was this perceptron from being correct.
     * Effectively, the above equation is the derivative of the Total Error function: 
     * ETotal = 0.5 * (expected_output - actual_output)^2
     */
    // NOCAB TODO: pass in a struct object into the cost func so you can provide paramater 
    // names. The problem is it's confusing label then recent output? Or output then label?
    // A data struct would remove this issue.

    return  p._costFunc(expectedOutput, p.recentOutput);
  }

  private double dOut_over_dNet(Perceptron p) {
    /**
     * dOutput
     * ------- = sigma'(this.recentOutput) = derivative of the activation function.
     *  dNet
     *
     * "How much of the output of this perceptron change with respect to the total
     *  net inputs?"
     *
     * NOTE: Some derivative functions are more preformant when passing in the recent output,
     *  but some require the recent net. That's why the derivative function accepts in 
     *  a Perceptron. The derivative function itself will have the choice of using
     *  Perceptron.recentNet or Perceptron.recentOutput.
     */
    return p._activatorFuncDerivative(p);
  }
#endregion Back Propagation Math

}; // public class Optimizer

}; // namespace NoNuNe