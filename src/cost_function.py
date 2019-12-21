

def compute_cost1(alpha, beta, time, distance):
    return alpha * time


def compute_cost2(alpha, beta, time, distance):
    return alpha * (time**beta)


def compute_cost3(alpha, beta, time, distance):
    return alpha * (time**beta) * distance

