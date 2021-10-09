using System.Collections;
using System;
using System.Collections.Generic;

namespace NoNuNe {
  
public class Layer : IEnumerable<Perceptron>
{

  public int LayerId { get; set; }

  private List<Perceptron> perceptrons = new List<Perceptron>();

  PerceptronFactory pFact = new PerceptronFactory();
  public Layer() {
    // Simple default constructor. 
  }

  public Layer(int layerId, int size) {
    this.LayerId = layerId;

    pFact.setActivatorFunc(PerceptronFactory.EActivationFunction.Sigmoid);
    for (int i = 0; i < Math.Abs(size); i++) {
      this.appendPerceptron(pFact.buildPerceptron(layerId: this.LayerId));
    }
  }

  public List<double> evaluate(List<double> inputs) {
    List<double> result = new List<double>(inputs.Count);
    for(int i = 0; i < inputs.Count; i++) {
      Perceptron p = this.perceptrons[i];
      result.Add(p.activationValue(inputs));
    }
    return result;
  }

  public void applyUpdateWeights() {
    foreach(Perceptron p in this.perceptrons) {
      p.applyUpdateWeights();
    }
  }

  public void armUpdateWeights(Layer leftLayer, List<double> expectedOutput) {
    // TODO: Consider growing the output layer to match expected output
    if (expectedOutput.Count != this.perceptrons.Count) {
      string errMsg = $"Invalid output layer backprop. Provided expectedOutput " +
       $"is NOT the same size as the number of perceptrons. " +
       $"\nexpectedOutput.Count: {expectedOutput.Count} != perceptrons.Count: {perceptrons.Count}";
      throw new ArgumentException(errMsg);
    }

    for(int i = 0; i < this.perceptrons.Count; i++) {
      Perceptron p = this.perceptrons[i];
      double expected = expectedOutput[i];

      p.armUpdateWeights(leftLayer, expected);
    }
  }

  public void armUpdateWeights(Layer leftLayer, Layer rightLayer) {
    foreach(Perceptron p in this.perceptrons) {
      p.armUpdateWeights(leftLayer, rightLayer);
    }
  }

  public void appendPerceptron(Perceptron p) {
    p.LayerId = this.LayerId;
    p.PerceptronId = this.perceptrons.Count;
    this.perceptrons.Add(p);
  }

  public Perceptron getPerceptron(int perceptronId) {
    if ((perceptronId < 0) || (perceptronId >= this.perceptrons.Count)) {
      throw new ArgumentException($"Invalid perceptronId: {perceptronId}");
    }

    return this.perceptrons[perceptronId];
  }

  public IEnumerator<Perceptron> GetEnumerator() {
    return this.perceptrons.GetEnumerator();
  }

  IEnumerator IEnumerable.GetEnumerator() {
    return this.GetEnumerator();
  }

}; // class Layer

} // namespace NoNuNe
