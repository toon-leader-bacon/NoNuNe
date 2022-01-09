using System;
using System.Collections.Generic;
using LightJson;

namespace NoNuNe {
public class Perceptron : INeuron {

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
  public List<double> getAllWeights() { return this._currentWeights; }

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
  public Func<Perceptron, double> _activatorFuncDerivative;

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
  private double _learningRate = 0.005d;
  public double learningRate { get { return this._learningRate; } }

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

  public Perceptron(double initialThreshold,
                    PerceptronFactory.EActivationFunction activationEnum,
                    PerceptronFactory.ECostFunction costFunction) {
    this._threshold = initialThreshold;
    this.activatorTypeEnum = activationEnum;
    this.activatorFunc = PerceptronFactory.getActivatorFunc(this.activatorTypeEnum);
    this._activatorFuncDerivative = PerceptronFactory.getDerivativeFunc(this.activatorTypeEnum);

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

  public double activatorFuncDerivative() {
    return this._activatorFuncDerivative(this);
  }

#region Backprop

  public double getWeight(int perceptronIndex) {
    if ((perceptronIndex < 0) || (perceptronIndex >= this._currentWeights.Count)) {
      throw new ArgumentException($"Invalid perceptron id: {perceptronIndex}");
    }
    return this._currentWeights[perceptronIndex];
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
  }

  public void armUpdateWeights(Layer leftLayer, double expectedOutput) {
    /**
     * Output layer update weights function. 
     * See _updateWeights(...) for the math.
     */
    this._newWeights = optimizer.calculateNewWeights(this, leftLayer, expectedOutput);
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
    this._activatorFuncDerivative = PerceptronFactory.getDerivativeFunc(this.activatorTypeEnum);

    // Cost function
    PerceptronFactory.ECostFunction newCostTypeEnum;
    enumParseSuccess = Enum.TryParse<PerceptronFactory.ECostFunction>(jo["CostTypeEnum"].AsString, out newCostTypeEnum);
    // If parsing fails, default is Cross Entropy :shrug:
    this.costTypeEnum = (enumParseSuccess) ? (newCostTypeEnum) : (PerceptronFactory.ECostFunction.CrossEntropy); 
    this.costFunc = PerceptronFactory.getCostFunc(this.costTypeEnum);

    this._threshold = jo["Threshold"];
    this._learningRate = jo["LearningRate"];
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
