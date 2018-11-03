class Answer(object):'''68. Text Justification'''
    def fullJustify(words, maxWidth):
        '''Use a queue. For each line, add words to the queue, including spaces as delimitor,
         while keeping count of number of chars. If the queue overflows,
          then add to the result and empty the line in the queue.
            O(n) time, O(ll) space , n is number of words, ll is length of line'''
        sentences = []
        num_char, sent_words = 0, []
        for word in words:
            num_char += len(word)
            sent_words.append(word)
            if num_char + len(sent_words) - 1 > maxWidth:
                num_char -= len(word)
                sent_words.pop()
                if len(sent_words) == 1: sent_words.append('') # make left justifiy if only one word
                avg_space, overflow_spaces = divmod((maxWidth - num_char), len(sent_words) - 1)
                for i in range(1, overflow_spaces + 1): sent_words[i] = ' ' + sent_words[i]
                sentences.append((' ' * avg_space).join(sent_words))
                num_char, sent_words = len(word), [word]
        sentences.append(' '.join(sent_words).ljust(maxWidth))
        return sentences