using System;
using System.Collections.Generic;
namespace NoNuNe {
public class Network
{

  private List<Layer> layers = new List<Layer>();

  public Network() {
    // Empty constructor
  }

  public Network(int size) {
    LayerFactory lf = new LayerFactory();
    for (int i = 0; i < Math.Abs(size); i++) {
      layers.Add(lf.buildLayer(layerId: i, 2));
    }
  }


  public void backprop(List<double> expectedOutput) {
    Layer outputLayer = this.layers[this.layers.Count - 1];
    Layer leftmostHidden = this.layers[this.layers.Count - 2];

    outputLayer.armUpdateWeights(leftmostHidden, expectedOutput);

    for(int i = this.layers.Count - 2; i > 0; i--) {
      Layer currentLayer = this.layers[i];
      Layer leftLayer = this.layers[i - 1];
      Layer rightLayer = this.layers[i + 1];
      currentLayer.armUpdateWeights(leftLayer, rightLayer);
    }

    for(int i = 1; i < this.layers.Count; i ++) {
      this.layers[i].applyUpdateWeights();
    }
  }

  public List<double> evaluate(List<double> input) {
    foreach(Layer l in this.layers) {
      // The output of a given layer becomes the input of the next
      
      input = l.evaluate(input);
    }

    // The output of the last layer is the output of the network.
    return input;
  }

  public void appendLayer(Layer l) {
    l.LayerId = this.layers.Count;
    this.layers.Add(l);
  }

  public Layer getLayer(int layerId) {
    if ((layerId < 0) || (layerId >= this.layers.Count)) {
      throw new ArgumentException($"Invalid layerID: {layerId}");
    }

    return this.layers[layerId];
  }

  public Perceptron GetPerceptron(int layerId, int perceptronId) {
    return this.getLayer(layerId).getPerceptron(perceptronId);
  }

}; // Class Network

} // Namespace NoNuNe
