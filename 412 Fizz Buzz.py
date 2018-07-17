class Answer(object):
'''412. Fizz Buzz'''
    def fizzBuzz(n):
        return [('Fizz' * (not i % 3) + 'Buzz' * (not i % 5)) or str(i) for i in range(1, n + 1)]