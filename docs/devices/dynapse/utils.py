        if isinstance(dest, list):
            return __get_tag(dest[0]) if dest else -1
        else:
            return dest.tag
