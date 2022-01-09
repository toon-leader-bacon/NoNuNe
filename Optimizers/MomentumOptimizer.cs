using System.Collections.Generic;

namespace NoNuNe {

public class MomentumOptimizer {

  public WeightDeltaMatrix weightDeltaLookup;
  public double momentumInfluence = 0.3d;
  public double gradientDescentInfluence;

  // Used during hidden layer backpropogation 
  private Dictionary<INeuron, double> lowercaseDeltaLookup = new Dictionary<INeuron, double>();

  public MomentumOptimizer(int historyLength = 10) {
    this.weightDeltaLookup = new WeightDeltaMatrix(historyLength);
    gradientDescentInfluence = 1d - momentumInfluence;
  }

#region Backprop

  public List<double> calculateNewWeights(INeuron p, Layer leftLayer, double expectedOutput) {
    /**
     * Output layer update weights function. 
     * See _updateWeights(...) for the math.
     */
    double lowercaseDelta = output_dETot_over_dOut(p, expectedOutput) * dOut_over_dNet(p);
    lowercaseDeltaLookup[p] = lowercaseDelta; // Used during subsequent hidden layer backprop
    return gradientDescentWithMomentum(p, leftLayer, lowercaseDelta);
  }

  public List<double> calculateNewWeights(INeuron p, Layer leftLayer, Layer rightLayer) {
    /**
     * Hidden layer update weights function. 
     * See _updateWeights(...) for the math.
     * NOTE: after all the layers have been "armed" to update their weights,
     *  you must still run the applyUpdateWeights for it to take effect.
     *
     * The arm and apply are two steps because leftern layers need the "old"
     * weights as part of the backpropagation algorithm. 
     */
    double lowercaseDelta = hidden_dETot_over_dOut(p, rightLayer) * dOut_over_dNet(p);
    lowercaseDeltaLookup[p] = lowercaseDelta; // Used during subsequent hidden layer backprop
    return gradientDescentWithMomentum(p, leftLayer, lowercaseDelta);
  }

#region Deep Math

 private double output_dETot_over_dOut(INeuron p, double expectedOutput) {
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
     * 
     */
    return p.recentOutput - expectedOutput;
  }

  private double hidden_dETot_over_dOut(INeuron p, Layer rightLayer) {
    /**              
     * dErrorTotal   dErr_Right0   dErr_Right1         dErr_RightI
     * ----------- = ----------- + ----------- + ... + ------------ = (lowerDelta0 * Weight0) + (lowerDelta1 * Weight1) + ... + (lowerDeltaI * WeightI)
     *   dOutput     dErr_Output   dErr_Output           dOutput
     *                                               
     *                                                     ||
     *      ++=============================================++
     *      ||
     *     
     * dErr_RightI   dErr_RightI   dNet_RightI
     * ----------- = ----------- * ----------- = lowercaseDelta_RightI * WeightI
     *   dOutput     dNet_RightI     dOutput
     *
     *                  ||             ||
     *     ++===========++             ++====================================++
     *     ||                                                                ||
     *                                                                       ||
     * dErr_RightI    dErr_RightI   dOut_RightI                              ||
     * ------------ = ----------- * ----------- = lowercaseDelta_RightI      ||
     * dNet_RightI    dOut_RightI   dNet_RightI                              ||
     *                                                                       ||
     *    ++=================================================================++
     *    ||
     *
     * dNet_RightI
     * ----------- = WeightI = The weight connecting this Perceptron to the rightI Perceptron
     *   dOutput
     *
     * 
     * dErrorTotal   
     * ----------- = (lowerDelta0 * Weight0) + (lowerDelta1 * Weight1) + ... + (lowerDeltaI * WeightI)
     *   dOutput
     *
     *
     * How far off was this perceptron from being correct
     */
    double result = 0d;
    foreach (INeuron rightP in rightLayer) {
      result += (lowercaseDeltaLookup[rightP] * rightP.getWeight(p.PerceptronId));
    }
    return result;
  }

  private double dOut_over_dNet(INeuron p) {
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
    return p.activatorFuncDerivative();
  }

  private List<double> gradientDescentWithMomentum(INeuron p, Layer leftLayer, double lowercaseDelta) {
    /**
     * Chain rule: 
     *  dETotal    dETotal   dOutput      dNet                         dNet
     * --------- = ------- * ------- * --------- = lowercaseDelta * ---------
     * dWeight_i   dOutput    dNet     dWeight_i                    dWeight_i
     *
     * New Weight_i = old_weight_i - (learningRate * (dETotal/ dWeight_i))
     * 
     * NOTE: This optimization function also includes momentum.
     *
     * Notice:
     *  dETotal   dOutput
     *  ------- * ------- = lowercaseDelta => The same value for every input weight_i
     *  dOutput    dNet 
     */
    List<double> result = new List<double>();
    for(int weightIndex = 0; weightIndex < p.getAllWeights().Count; weightIndex++) {
      // Gradient Descent 
      double dETot_over_dWeight = lowercaseDelta * dNet_over_dWeight(leftLayer, weightIndex);
      double weightDelta = (p.learningRate * dETot_over_dWeight);

      // Weight delta momentum 
      // TODO: The toArray() call is really inefficient :( 
      double momentum = average(weightDeltaLookup.getRecentWeightDeltas(p, weightIndex).ToArray());
      weightDeltaLookup.appendWeightDelta(p, weightIndex, weightDelta);

      // Calculate new weight 
      double totalDelta = (weightDelta * gradientDescentInfluence) - (momentum * momentumInfluence);
      result.Add(p.getAllWeights()[weightIndex] - totalDelta);
    }
    return result;
  }

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

#endregion Deep Math

#endregion Backprop

  public static double average(ICollection<double> weights) {
    if(weights.Count == 0) { return 0d; }
    double sum = 0d;
    foreach(double w in weights) {
      sum += w;
    }
    return sum / weights.Count;
  }

}; // public class MomentumOptimizer

}; // namespace NoNuNe