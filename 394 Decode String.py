class Answer(object):
'''394. Decode String'''
    def decodeString(s):
        stack = []
        str_queue = []
        def add_to_queue(str_queue, stack):
            stack.append(''.join(str_queue))
            del str_queue[:]
            last = []
            while stack and not stack[-1].isdigit(): last.append(stack.pop())
            if last: stack.append(''.join(list(reversed(last))))
        for char in s:
            if '[' == char:
                add_to_queue(str_queue, stack)
            elif ']' == char:
                add_to_queue(str_queue, stack)
                stack.append(stack.pop() * int(stack.pop()))
            elif str_queue and str_queue[-1].isdigit() != char.isdigit():
                add_to_queue(str_queue, stack)
                str_queue.append(char)
            else: str_queue.append(char)
        stack.append(''.join(str_queue))
        return ''.join(stack)