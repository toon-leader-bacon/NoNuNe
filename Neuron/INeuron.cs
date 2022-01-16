using LightJson;
using System.Collections.Generic;

namespace NoNuNe {

public interface INeuron : JsonConvertible {

  /**
   * The unique identifier for the layer this perceptron is part of.
   * Typically, LayerIds are the index of the layer with the input 
   * layer having and id = 0
   */
  int LayerId { get; set; }

  /** 
   * The unique identifier for this perceptron in a layer.
   * Typically, PerceptonrID is the index of the perceptron
   * with the "topmost" perceptron having an id = 0 
   */
  int PerceptronId { get; set; }
  
  /**
   * The most recent Output value calculated by this perceptron.
   * Set during forward propagation, used during back propagation.
   * This value SHOULD only be set by the activationValue(...) function.
   * output = activatorFunc(net) 
   *        = activatorFunc([inputs] * [currentWeights] - threshold)
   */
  double recentOutput { get; } // In textbook "a"

  /**
   * A value that represents how quickly this perceptron should adjust
   * itself during back propagation. Typically between [0.1, 0.01]
   */
  double learningRate { get; }

  /**
   * Given a vector of input values, evaluate them and then
   * use the specified activator function.
   * a = sigma(z)
   * 
   * a = output of this function (activation value)
   * sigma = activation function
   * z = output of evaluate(xInputs) function
   */
  double activationValue(List<double> xInputs); 


  /**
   * Hidden layer update weights function. 
   * See _updateWeights(...) for the math.
   * NOTE: after all the layers have been "armed" to update their weights,
   *  you must still run the applyUpdateWeights for it to take effect.
   *
   * The arm and apply are two steps because leftern layers need the "old"
   * weights as part of the backpropagation algorithm. 
   */
  void armUpdateWeights(Layer leftLayer, Layer rightLayer);

  /**
   * Output layer update weights function. 
   * See _updateWeights(...) for the math.
   */
  void armUpdateWeights(Layer leftLayer, double expectedOutput);

  /**
   * Get the value of the incoming weight to this perceptron. The
   * specific incoming weight is specified by the provided perceptron ID.
   */
  double getWeight(int perceptronIndex);

  /**
   * The weights for each incoming value for this perceptron.
   * This list should match the length of the incoming inputs
   * in the activationValue(...) function.
   * net = [inputs] * [currentWeights] - threshold
   */
  List<double> getAllWeights();

  double activatorFuncDerivative();

}; // public interface INeuron

}; // namespace NoNuNe
