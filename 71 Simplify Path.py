class Answer(object):'''71. Simplify Path'''
    simplified_path = []
    for folder in path.split('/'):
        if folder == '..':
            simplified_path.pop()
        else:
            simplified_path.append(folder)
    simplified_path = '/'.join(simplified_path)