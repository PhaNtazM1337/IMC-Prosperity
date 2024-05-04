def calc_payoff(lo, hi):
    payoff = 0
    for i in range(900, 1001):
        if i<lo:
            payoff += (1000-lo) * (i-900)
        elif i<hi:
            payoff += (1000-hi) * (i-900)
    return payoff


max_expected_payoff = -1
opt_param = (-1, -1)
for i in range(900,1001):
    for j in range(i,1001):
        expected_payoff = calc_payoff(i, j)
        if expected_payoff > max_expected_payoff:
            max_expected_payoff = expected_payoff
            opt_param = (i, j)
print(opt_param)

