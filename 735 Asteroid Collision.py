class Answer(object):
'''735. Asteroid Collision'''
    def asteroidCollision(asteroids):
        stack = []
        while asteroids:
            stack.append(asteroids.pop())
            while len(stack) >= 2 and stack[-1] > 0 and stack[-2] < 0:
                a, b = stack.pop(), stack.pop()
                if abs(a) > abs(b):
                    stack.append(a)
                elif abs(a) < abs(b):
                    stack.append(b)
        return stack[::-1]