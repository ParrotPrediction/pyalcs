class Configuration:
    def __init__(self,
                 classifier_length: int) -> None:
        self.classifier_length = classifier_length

        # encoder bits
        # wildcard: [0.0, 1.0]
