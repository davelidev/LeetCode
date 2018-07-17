class Answer(object):
'''204. Count Primes'''
    def countPrimes(n):
        if n <= 2: return 0
        primes = [True] * (n)
        primes[0] = primes[1] = False
        for i in range(2, n):
            if primes[i]:
                for j in range(i, n, i):
                    if j != i: primes[j] = False
        return primes.count(True)