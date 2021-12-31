using System;
using System.Collections.Generic;

namespace NoNuNe {

public class WeightMatrix {

  List<List<double>> recent_eTot_over_dWeight = new List<List<double>>();

  public void appendWeight(int weightIndex, double newWeightDelta) {
    assertValidWeightIndex(weightIndex);
    recent_eTot_over_dWeight.Add(newWeightDelta);
  }

  public List<double> getRecentForWeight(int weightIndex, int howManyRecent) {
    /**
     * For a given weight index, get a sub list no larger than the provided 
     * howManyRecent count. This sub list represents the most recent weight
     * changes to a given weight.
     *
     * @param weightIndex the identifier of the weight
     * @param howManyRecent the max length of the returned list. If this param
     *  is larger than the weight history, the entire history will be returned.
     * @returns A list of the most recent changes applied to a given weight 
     */
    assertValidWeightIndex(weightIndex);
    List<double> allWeights = getAllWeights(weightIndex);

    // Normalize the data in case someone puts in a negative number
    howManyRecent = Math.Max(howManyRecent, 0);

    int recentCount = Math.Min(allWeights.Count, howManyRecent);
    int firstIndex = allWeights.Count - recentCount;
    return allWeights.GetRange(firstIndex, recentCount);
  }

  public List<double> getAllWeights(int weightIndex) {
    assertValidWeightIndex(weightIndex);

    return recent_eTot_over_dWeight[weightIndex];
  }

  private void assertValidWeightIndex(int weightIndex) {
    if((weightIndex < 0) || (weightIndex >= recent_eTot_over_dWeight.Count) ) {
      // If the provided index is invalid
      throw new ArgumentException(
        $"Invalid weight index: '{weightIndex}' for weight list of length: {recent_eTot_over_dWeight.Count}");
    }
  }

}; // public class Optimizer

}; // namespace NoNuNe