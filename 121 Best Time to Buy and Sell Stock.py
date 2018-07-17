class Answer(object):
'''121. Best Time to Buy and Sell Stock'''
    def maxProfit(prices):
        max_profit = max_so_far = 0
        for price in reversed(prices):
            max_profit = max(max_so_far - price, max_profit)
            max_so_far = max(max_so_far, price)
        return max_profit