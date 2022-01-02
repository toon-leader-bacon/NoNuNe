using System;
using System.Linq;
using System.Collections.Generic;

namespace NoNuNe {

public class WeightDeltaMatrix {

  Dictionary<int, List<double>> recentWeightDeltas = new Dictionary<int, List<double>>();

  public void appendWeightDelta(Perceptron p, int weightId, double newWeightDelta) {
    this.appendWeightDelta(p.LayerId, p.PerceptronId, weightId, newWeightDelta);
  }   

  public void appendWeightDelta(int LayerId, int PerceptronId, int weightId, double newWeightDelta) {
    int trueHash = NocabHashUtility.generateHash(LayerId, PerceptronId, weightId);
    if (!recentWeightDeltas.ContainsKey(trueHash)) {
      // If the provided true has is new
      recentWeightDeltas.Add(trueHash, new List<double>());
    }

    recentWeightDeltas[trueHash].Add(newWeightDelta);
  }

  public List<double> getRecentWeightDeltas(Perceptron p, int weightId, int howMany) {
    return getRecentWeightDeltas(p.LayerId, p.PerceptronId, weightId, howMany);
  }

  public List<double> getRecentWeightDeltas(int LayerId, int PerceptronId, int weightId, int howMany) {
    int trueHash = NocabHashUtility.generateHash(LayerId, PerceptronId, weightId);
    if(!recentWeightDeltas.ContainsKey(trueHash)) {
      // If the provided true hash is new
      return new List<double>();
    }

    List<double> allWeightDeltas = recentWeightDeltas[trueHash];
    howMany = Math.Abs(howMany);
    if (howMany >= allWeightDeltas.Count) {
      return allWeightDeltas;
    }

    int startIndex = allWeightDeltas.Count - howMany;
    return allWeightDeltas.GetRange(startIndex, howMany);
  }

}; // public class MomentumOptimizer

}; // namespace NoNuNe