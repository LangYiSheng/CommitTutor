
def should_request_advice(score, threshold=0.6):
    return score >= threshold


__all__ = ["should_request_advice"]
