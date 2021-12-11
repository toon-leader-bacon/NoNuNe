using System.Collections;
using System;
using System.Collections.Generic;
using LightJson;

namespace NoNuNe {
  
public class Layer : IEnumerable<Perceptron>, JsonConvertible
{

  public int LayerId { get; set; }

  private List<Perceptron> perceptrons = new List<Perceptron>();
  public int Count { get {return perceptrons.Count; } }


  PerceptronFactory pFact = new PerceptronFactory();
  public Layer() {
    // Simple default constructor. 
  }

  public Layer(JsonObject jo) {
    this.loadJson(jo);
  }

  public List<double> evaluate(List<double> inputs) {
    List<double> result = new List<double>(perceptrons.Count);
    for(int i = 0; i < perceptrons.Count; i++) {
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

#region JsonConvertible

  public const string MY_JSON_TYPE = "Layer_1.0";

  public string myJsonType() {
    return MY_JSON_TYPE;
  }

  public JsonObject toJson() {
    JsonObject result = JsonUtilitiesNocab.initJson(MY_JSON_TYPE);
    result["LayerId"] = this.LayerId;

    JsonArray jsonPerceps = new JsonArray();
    foreach(Perceptron p in this.perceptrons) {
      jsonPerceps.Add(p.toJson());
    }
    result["Perceptrons"] = jsonPerceps;

    return result;
  }

  public void loadJson(JsonObject jo) {
    JsonUtilitiesNocab.assertValidJson(jo, MY_JSON_TYPE);
    this.LayerId = jo["LayerId"];
    
    this.perceptrons = new List<Perceptron>();
    foreach(JsonObject jsonPercep in jo["Perceptrons"].AsJsonArray) {
      this.perceptrons.Add(new Perceptron(jsonPercep));
    }
  }

#endregion JsonConvertible

}; // class Layer

} // namespace NoNuNe
