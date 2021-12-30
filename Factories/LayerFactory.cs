using System;
namespace NoNuNe {
  
public class LayerFactory : PerceptronFactory{

  public Layer buildLayer(int layerId = 0,
                          int size = 0) {
    return this.buildLayer(layerId: layerId, size: size, EActivationFunction.Sigmoid);
  }


  public Layer buildLayer(int layerId,
                          int size,
                          EActivationFunction perceptronActivation) {
    return this.buildLayer(
      layerId: layerId, 
      size: size, 
      perceptronActivation: perceptronActivation, 
      costFunction: ECostFunction.Linear);  // Default
  }

  public Layer buildLayer(int layerId,
                          int size,
                          EActivationFunction perceptronActivation, 
                          ECostFunction costFunction) {
    Layer result = new Layer();
    result.LayerId = layerId;
    
    this.setActivatorFunc(perceptronActivation);
    size = Math.Abs(size);
    for(int i = 0; i < size; i++) {
      Perceptron p = this.buildPerceptron(
        layerId: layerId,
        perceptronId: i
      );
      result.appendPerceptron(p);
    }

    return result;
  }

#region Custom Layer building

  // The layer that is "on deck" in the process of being built manually
  private Layer layerInProgress = new Layer();

  public Layer compileLayer() {
    /**
      * Returns and then resets the current list that is being built up manually.
      * Think of this function like a "println" function: Take whatever
      * is in the buffer, return it and clear the buffer.
      * @returns The layer that has been previously built up.
      */
    Layer result = layerInProgress;
    layerInProgress = new Layer();
    return result;
  }

  public Layer clearLayer() {
    return compileLayer();
  }

  public void appendPerceptron(Perceptron p) {
    /**
      * Adds the provided perceptron to the currently manually built
      * layer. 
      * @param p the perceptron to add
      */
    layerInProgress.appendPerceptron(p);
  }

#endregion Custom Layer building

}; // class LayerFactory

}; // namespace NoNuNe
