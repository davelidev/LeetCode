class Answer(object):
'''806. Number of Lines To Write String'''
    def numberOfLines(widths, S):
        num_lines = line_width = 0
        for i, char in enumerate(S):
            width = widths[ord(char) - ord('a')]
            if not num_lines or line_width + width > 100:
                line_width = width
                num_lines += 1
            else:
                line_width += width
        return (num_lines, line_width)