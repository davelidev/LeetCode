class Answer(object):
'''520. Detect Capital'''
    def detectCapitalUse(word):
        is_chars_upper = [(char == char.upper()) for char in reversed(word)]
        first_char_upper = is_chars_upper.pop()
        return first_char_upper and all(is_chars_upper) or not any(is_chars_upper)