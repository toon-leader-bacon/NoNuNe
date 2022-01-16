using System.Collections.Generic;
using LightJson;


namespace NoNuNe {
public class IdentityNeuron : INeuron {

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

  private double _recentOutput;
  public double recentOutput { get { return _recentOutput; } }

  private double _learningRate;
  public double learningRate { get { return _learningRate;} }

  public double getWeight(int perceptronIndex) {
    return 1;
  }

  public double activationValue(List<double> xInputs) {
    if(this.PerceptronId >= xInputs.Count) {
      // TODO: Consider returning noise instead of 0?
      return 0;
    }
    return xInputs[PerceptronId];
  }
  
  public void armUpdateWeights(Layer leftLayer, Layer rightLayer) {
    // Identity neuron does not change during back propagation
    return;
  }

  public void armUpdateWeights(Layer leftLayer, double expectedOutput) {
    // Identity neuron does not change during back propagation
    return;
  }


  public List<double> getAllWeights() {
    return new List<double>{1};
  }

  public double activatorFuncDerivative() {
    return 0;
  }

 #region Jsonable interface

  
  public const string MY_JSON_TYPE = "IdentityNeuron_1.0";

  public string myJsonType() {
    return MY_JSON_TYPE;
  }

  public JsonObject toJson() {
    JsonObject result = JsonUtilitiesNocab.initJson(MY_JSON_TYPE);
    result["LayerId"] = this.LayerId;
    result["PerceptronId"] = this.PerceptronId;
    result["LearningRate"] = this.learningRate;
    return result;
  }

  public void loadJson(JsonObject jo) {
    JsonUtilitiesNocab.assertValidJson(jo, MY_JSON_TYPE);

    this.LayerId = jo["LayerId"];
    this.PerceptronId = jo["PerceptronId"];
    this._learningRate = jo["LearningRate"];
  }

 #endregion Jsonable interface

}; // public class IdentityNeuron

}; // namespace NoNuNe
