using System.Security.Claims;
using System.Collections.Generic;

public class DataPoint {
  public List<double> input = new List<double>();
  public List<double> expectedOutput = new List<double>();

  public DataPoint(List<double> input, List<double> expectedOutput) {
    this.input = input;
    this.expectedOutput = expectedOutput;
  }

  public DataPoint() {}
}