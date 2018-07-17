class Answer(object):
'''122. Best Time to Buy and Sell Stock II'''
    def maxProfit(prices):
        profit = 0
        for i in range(1, len(prices)):
            profit += prices[i] - prices[i - 1] if prices[i] > prices[i - 1] else 0
        return profit