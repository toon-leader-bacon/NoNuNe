using System;
using System.Collections.Generic;

namespace NoNuNe {

public class WeightDeltaMatrix {


  Dictionary<int, Queue<double>> recentWeightDeltas = new Dictionary<int, Queue<double>>();


  private int historyLength = 100;

  public WeightDeltaMatrix(int maintainHistoryLength = 100) {
    this.historyLength = maintainHistoryLength;
  }

  public void appendWeightDelta(INeuron p, int weightId, double newWeightDelta) {
    this.appendWeightDelta(p.LayerId, p.PerceptronId, weightId, newWeightDelta);
  }   

  public void appendWeightDelta(int LayerId, int PerceptronId, int weightId, double newWeightDelta) {
    int trueHash = NocabHashUtility.generateHash(LayerId, PerceptronId, weightId);
    if (!recentWeightDeltas.ContainsKey(trueHash)) {
      // If the provided true has is new
      recentWeightDeltas.Add(trueHash, new Queue<double>());
    }
    
    Queue<double> targetDeltas = recentWeightDeltas[trueHash];
    while(targetDeltas.Count >= this.historyLength) {
      targetDeltas.Dequeue();
    }

    targetDeltas.Enqueue(newWeightDelta);
  }

  public Queue<double> getRecentWeightDeltas(INeuron p, int weightId) {
    return getRecentWeightDeltas(p.LayerId, p.PerceptronId, weightId);
  }

  public Queue<double> getRecentWeightDeltas(int LayerId, int PerceptronId, int weightId) {
    int trueHash = NocabHashUtility.generateHash(LayerId, PerceptronId, weightId);
    if(!recentWeightDeltas.ContainsKey(trueHash)) {
      // If the provided true hash is new
      return new Queue<double>();
    }

    return recentWeightDeltas[trueHash];
  }

}; // public class MomentumOptimizer

}; // namespace NoNuNe