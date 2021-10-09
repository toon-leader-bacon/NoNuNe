using System.Collections.Generic;
using System;
using System.Collections.Generic;

namespace NoNuNe
{
  class Program
  {
    static void Main(string[] args) {
      Layer l0 = new Layer(0, 4);  // Input layer
      Layer l1 = new Layer(1, 16);
      Layer l2 = new Layer(2, 32);
      Layer l3 = new Layer(3, 10); // Output layer

      Network n = new Network(0);
      n.appendLayer(l0);
      n.appendLayer(l1);
      n.appendLayer(l2);
      n.appendLayer(l3);
      
    }
  }
}
