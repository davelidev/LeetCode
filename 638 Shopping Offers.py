class Answer(object):
'''638. Shopping Offers'''
    def shoppingOffers(price, special, needs):
        def dfs(total=0, idx=0):
            if idx == len(special):
                return total + sum(need * price[i] for i, need in enumerate(needs))
            cost = []
            n, quan = len(needs), 0
            while all(needs[i] - quan * special[idx][i] >= 0 for i in range(n)):
                for i in range(n): needs[i] = needs[i] - quan * special[idx][i]
                cost.append(dfs(total + special[idx][-1] * quan, idx + 1))
                for i in range(n): needs[i] = needs[i] + quan * special[idx][i]
                quan += 1
            return min(c for c in cost)
        return dfs()