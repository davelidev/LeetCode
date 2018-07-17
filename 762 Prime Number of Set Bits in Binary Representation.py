class Answer(object):
'''762. Prime Number of Set Bits in Binary Representation'''
    def countPrimeSetBits(L, R):
        primes = {2, 3, 5, 7, 11, 13, 17, 19}
        return sum((bin(i).count('1') in primes) for i in range(L, R + 1))