
class DataModel:
    def __init__(self, inputs, expected_output):
        self.inputs = inputs
        self.expected_output = expected_output

    def get_inputs(self):
        return self.inputs

    def get_expected_outputs(self):
        return self.expected_output
