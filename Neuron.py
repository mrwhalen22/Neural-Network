class Neuron:
    
    def __init__(self, bias : float, inputs : dict, output : float) -> None:
        self.bias = bias
        self.inputs = inputs
        self.outputs = output

    def calculate_output(self) -> float:
        self.output = 0
        for x in range(0, len(self.inputs["values"])):
            self.output += self.inputs["values"][x] * self.inputs["weights"][x] + self.bias
        return self.output

    