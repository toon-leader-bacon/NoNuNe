using System;
using System.Collections.Generic;
using LightJson;

namespace NoNuNe {
public class Perceptron : JsonConvertible {

  // TODO: Should this stay static? 
  public static MomentumOptimizer optimizer = new MomentumOptimizer();

  /**
   * The unique identifier for the layer this perceptron is part of.
   * Typically, LayerIds are the index of the layer with the input 
   * layer having and id = 0
   */
  public int LayerId { get; set; }

  /** 
   * The unique identifier for this perceptron in a layer.
   * Typically, PerceptonrID is the index of the perceptron
   * with the "topmost" perceptron having an id = 0 
   */
  public int PerceptronId { get; set; }


  /**
   * The weights for each incoming value for this perceptron.
   * This list should match the length of the incoming inputs
   * in the activationValue(...) function.
   * net = [inputs] * [currentWeights] - threshold
   */
  public List<double> _currentWeights = new List<double>();

  /**
   * The new weights this perceptron should have once 
   * backpropagation has been finished. We save this till
   * the end of backprop because other layers need the "old"
   * weights as part of their backprop calculations.
   */
  private List<double> _newWeights = new List<double>();


  private PerceptronFactory.EActivationFunction activatorTypeEnum;

  /**
   * The Activator Function is used to convert the net value to 
   * the output value.
   * This Func type takes in a the dot product of the input values
   * times the current weights, minus the threshold, also known as
   * the Net value. It returns the Output value of this perceptron.
   * output = activatorFunc(net) 
   *        = activatorFunc([inputs] * [currentWeights] - threshold)
   */
  public Func<double, double> activatorFunc;

  /**
   * The Activator Function's Derivative is used during back 
   * propagation. 
   */
  public Func<Perceptron, double> activatorFuncDerivative;

  private PerceptronFactory.ECostFunction costTypeEnum;
  public Func<double, double, double> costFunc;


  /**
   * The Threshold is a POSITIVE value that represents the
   * perceptrons base reluctance to activate itself. This value
   * SHOULD always be positive. Some textbooks prefer to use 
   * the term Bias instead of Threshold. Bias = -Threshold
   */
  private double _threshold;
  public double Threshold { get { return _threshold; } }
  public double Bias { get { return -_threshold; } }

  /**
   * The most recent Net value calculated by this perceptron.
   * Set during forward propagation, used during back propagation.
   * This value SHOULD only be set by the _evaluate(...) function.
   * net = [inputs] * [currentWeights] - threshold
   */
  private double _recentNet;
  public double RecentNet { get { return this._recentNet; } }

  /**
   * The most recent Output value calculated by this perceptron.
   * Set during forward propagation, used during back propagation.
   * This value SHOULD only be set by the activationValue(...) function.
   * output = activatorFunc(net) 
   *        = activatorFunc([inputs] * [currentWeights] - threshold)
   */
  private double _recentOutput;
  public double recentOutput { get { return this._recentOutput; } } // In textbook "a"

  /**
   * A value that represents how quickly this perceptron should adjust
   * itself during back propagation. Typically between [0.1, 0.01]
   */
  public double learningRate = 0.005d;

  /**
   * A value used during back propagation. It's convenient to calculate 
   * it once and save the value here for reuse.
   * The value SHOULD only be set in the armUpdateWeights(...) functions.
   * This value is equal
   *  dETotal   dOutput
   *  ------- * ------- = lowercaseDelta
   *  dOutput    dNet 
   */
  private double lowercaseDelta;

  // public Perceptron(double initialThreshold,
  //                   Func<double, double> activatorFunc,
  //                   Func<Perceptron, double> activatorFuncDerivative,
  //                   Func<double, double, double> costFunction) {
  //   this._threshold = initialThreshold;
  //   this.activatorFunc = activatorFunc;
  //   this.activatorFuncDerivative = activatorFuncDerivative;

  //   // The weights will be populated "just in time" in the _evaluate() function
  //   // using the PerceptronFactory.randomWeight() function
  //   // Because the perceptron doesn't know how many inputs it's getting from
  //   // the layer before it. 
  //   this._currentWeights = new List<double>();
  // }

  public Perceptron(double initialThreshold,
                    PerceptronFactory.EActivationFunction activationEnum,
                    PerceptronFactory.ECostFunction costFunction) {
    this._threshold = initialThreshold;
    this.activatorTypeEnum = activationEnum;
    this.activatorFunc = PerceptronFactory.getActivatorFunc(this.activatorTypeEnum);
    this.activatorFuncDerivative = PerceptronFactory.getDerivativeFunc(this.activatorTypeEnum);

    this.costTypeEnum = costFunction;
    this.costFunc = PerceptronFactory.getCostFunc(this.costTypeEnum);
  }

  public Perceptron(JsonObject jo) {
    this._currentWeights = new List<double>();
    this._newWeights = new List<double>();
    this.loadJson(jo);
  }

  public double _evaluate(List<double> xInputs) {
    /**
     * Calculate the dot product between the provided xInputs and this.ms
     * then add the b value.
     * Basically, net = (m * x) + b
     * 
     * net = simple output from this function
     * m = vector of weights
     * x = vector of input values
     * b = -threshold
     */
    if (xInputs.Count > this._currentWeights.Count) {
      // If there are more inputs than weights
      // Add new random weights for each input.
      int delta = xInputs.Count - this._currentWeights.Count;
      for (int i = 0;  i < delta; i++) {
        this._currentWeights.Add(PerceptronFactory.randomWeight());
      }
    }

    double sum = 0.0d;
    // Sum all the weighted inputs
    for(int i = 0; i < xInputs.Count; i++) {
      double weight = this._currentWeights[i];
      double input = xInputs[i];
      sum += weight * input;
    }
    this._recentNet = sum - Threshold;
    return this._recentNet;
  }

  public double activationValue(List<double> xInputs) {
    /**
     * Given a vector of input values, evaluate them and then
     * use the specified activator function.
     * a = sigma(z)
     * 
     * a = output of this function (activation value)
     * sigma = activation function
     * z = output of evaluate(xInputs) function
     */
    this._recentNet = this._evaluate(xInputs); 
    this._recentOutput = this.activatorFunc(this._recentNet);
    return this.recentOutput;
  }

#region Backprop

  public double getWeight(int perceptronIndex) {
    if ((perceptronIndex < 0) || (perceptronIndex >= this._currentWeights.Count)) {
      throw new ArgumentException($"Invalid perceptron id: {perceptronIndex}");
    }
    return this._currentWeights[perceptronIndex];
  }


  private double output_dETot_over_dOut(double expectedOutput) {
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
     * TODO: Consider using a different error calculation function? 
     * TODO: And therefore different derivative here
     */
    return this.recentOutput - expectedOutput;
  }

  private double dOut_over_dNet() {
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
    return activatorFuncDerivative(this);
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

  private double hidden_dETot_over_dOut(Layer rightLayer) {
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
    foreach (Perceptron p in rightLayer) {
      result += (p.lowercaseDelta * p.getWeight(this.PerceptronId));
    }
    return result;
  }

  private void _updateWeights(Layer leftLayer, double lowercaseDelta) {
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
    this.lowercaseDelta = lowercaseDelta; // TODO: Some more elegant way to deal with passing a class variable
    this._newWeights = new List<double>(this._currentWeights.Count);
    for(int i = 0; i < this._currentWeights.Count; i++) {
      double dETot_over_dWeight = this.lowercaseDelta * dNet_over_dWeight(leftLayer, i);
      double newWeight = this._currentWeights[i] - (this.learningRate * dETot_over_dWeight);

      this._newWeights.Add(newWeight);
    }
  }

  public void armUpdateWeights(Layer leftLayer, Layer rightLayer) {
    /**
     * Hidden layer update weights function. 
     * See _updateWeights(...) for the math.
     * NOTE: after all the layers have been "armed" to update their weights,
     *  you must still run the applyUpdateWeights for it to take effect.
     *
     * The arm and apply are two steps because leftern layers need the "old"
     * weights as part of the backpropagation algorithm. 
     */
    this._newWeights = optimizer.calculateNewWeights(this, leftLayer, rightLayer);
    // this.lowercaseDelta = hidden_dETot_over_dOut(rightLayer) * this.dOut_over_dNet();
    // _updateWeights(leftLayer, this.lowercaseDelta);
  }

  public void armUpdateWeights(Layer leftLayer, double expectedOutput) {
    /**
     * Output layer update weights function. 
     * See _updateWeights(...) for the math.
     */
    this._newWeights = optimizer.calculateNewWeights(this, leftLayer, expectedOutput);
    // this.lowercaseDelta = this.output_dETot_over_dOut(expectedOutput) * this.dOut_over_dNet();
    // _updateWeights(leftLayer, this.lowercaseDelta);
  }

  public void applyUpdateWeights() {
    this._currentWeights = this._newWeights;
  }

#endregion

#region JsonConvertable

  public const string MY_JSON_TYPE = "Perceptron_1.0";

  public string myJsonType() {
    return MY_JSON_TYPE;
  }

  public JsonObject toJson() {
    JsonObject result = JsonUtilitiesNocab.initJson(MY_JSON_TYPE);
    result["LayerId"] = this.LayerId;
    result["PerceptronId"] = this.PerceptronId;

    JsonArray curWeights = new JsonArray();
    foreach(double weight in _currentWeights) {
      curWeights.Add(weight);
    }
    result["CurrentWeights"] = curWeights;
    result["ActivatorTypeEnum"] = this.activatorTypeEnum.ToString();
    result["CostTypeEnum"] = this.costTypeEnum.ToString();

    result["Threshold"] = this.Threshold;
    result["LearningRate"] = this.learningRate;

    return result;
  }

  public void loadJson(JsonObject jo) {
    JsonUtilitiesNocab.assertValidJson(jo, MY_JSON_TYPE);

    this.LayerId = jo["LayerId"];
    this.PerceptronId = jo["PerceptronId"];
    
    this._currentWeights = new List<double>();
    foreach(JsonValue jv in jo["CurrentWeights"].AsJsonArray){
      this._currentWeights.Add(jv.AsNumber);
    }

    this._newWeights = new List<double>();

    // Activator functions
    PerceptronFactory.EActivationFunction newActivatorTypeEnum;
    bool enumParseSuccess = Enum.TryParse<PerceptronFactory.EActivationFunction>(jo["ActivatorTypeEnum"].AsString, out newActivatorTypeEnum);
    // If parse fails, default is ReLU
    this.activatorTypeEnum = (enumParseSuccess) ? newActivatorTypeEnum : PerceptronFactory.EActivationFunction.ReLU;
    this.activatorFunc = PerceptronFactory.getActivatorFunc(this.activatorTypeEnum);
    this.activatorFuncDerivative = PerceptronFactory.getDerivativeFunc(this.activatorTypeEnum);

    // Cost function
    PerceptronFactory.ECostFunction newCostTypeEnum;
    enumParseSuccess = Enum.TryParse<PerceptronFactory.ECostFunction>(jo["CostTypeEnum"].AsString, out newCostTypeEnum);
    // If parsing fails, default is Cross Entropy :shrug:
    this.costTypeEnum = (enumParseSuccess) ? (newCostTypeEnum) : (PerceptronFactory.ECostFunction.CrossEntropy); 
    this.costFunc = PerceptronFactory.getCostFunc(this.costTypeEnum);

    this._threshold = jo["Threshold"];
    this.learningRate = jo["LearningRate"];
  }

    
#endregion JsonConvertable

  public override bool Equals(object obj) {
    return obj is Perceptron perceptron &&
           LayerId == perceptron.LayerId &&
           PerceptronId == perceptron.PerceptronId;
  }

  public override int GetHashCode() {
    return NocabHashUtility.generateHash(LayerId, PerceptronId);
  }

}
} // Namespace NoNuNe
