class Answer(object):'''925. Long Pressed Name'''
    def isLongPressedName(name, typed):
        def is_long_pressed(name, typed):
            if not name and not typed:
                return True
            elif not name or not typed:
                return False
            elif name[0] != typed[0]:
                return False
            else:
                n_dif, t_dif = len(name), len(typed)
                new_name, new_typed = name.lstrip(name[0]), typed.lstrip(typed[0])
                n_dif -= len(new_name)
                t_dif -= len(new_typed)
                return n_dif <= t_dif and is_long_pressed(new_name, new_typed)
        return is_long_pressed(name, typed)