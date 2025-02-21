alpha = 0.01
def smooth(prbs):
  V = prbs.size(-1)
  return (prbs + alpha) / (1 + V * alpha)