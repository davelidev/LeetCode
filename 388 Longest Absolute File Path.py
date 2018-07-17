class Answer(object):
'''388. Longest Absolute File Path'''
    def lengthLongestPath(input_file_sys):
        input_file_sys = input_file_sys.replace('    ', '	')
        input_file_sys = input_file_sys.split('
')
        input_file_sys = map(lambda x: [len(x) - len(x.lstrip('	')), x.lstrip('	')], input_file_sys)
        cur_dir = []
        longest_path = ''
        for lvl, file_name in input_file_sys:
            if len(cur_dir) <= lvl:
                cur_dir.append(file_name)
            else:
                cur_dir[lvl] = file_name
                while len(cur_dir) > lvl + 1:
                    cur_dir.pop()
            print '/'.join(cur_dir)
            if '.' in cur_dir[-1] and len(cur_dir) - 1 + sum(map(lambda x: len(x), cur_dir)):
                longest_path = '/'.join(cur_dir)
        return len(longest_path)