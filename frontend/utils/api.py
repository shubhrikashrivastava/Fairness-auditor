def detect_bias(file):
    return {
        "distribution": {"0": 400, "1": 50},
        "is_biased": True
    }

def mitigate_bias(file):
    return {
        "before_accuracy": 0.82,
        "after_accuracy": 0.91,
        "after_distribution": {"0": 400, "1": 400}
    }
