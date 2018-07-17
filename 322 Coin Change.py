class Answer(object):
'''322. Coin Change'''
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, len(dp)):
        for coin in coins:
            prev_idx = i - coin
            if prev_idx >= 0:
                dp[i] = min(dp[prev_idx] + 1, dp[i])
    res = dp[amount] if type(dp[amount]) == int else -1