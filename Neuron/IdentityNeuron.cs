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



}; // public class IdentityNeuron

}; // namespace NoNuNe
