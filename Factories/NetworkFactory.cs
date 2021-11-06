using System;

namespace NoNuNe {
  
public class NetworkFactory : LayerFactory {

  public Network buildNetwork(int numberOfLayers) {
    Network result = new Network();

    numberOfLayers = Math.Abs(numberOfLayers);
    for (int i = 0; i < numberOfLayers; i++) {
      result.appendLayer(buildLayer());
    }

    return result;
  }

}; // class NetworkFactory

}; // namespace NoNuNe
