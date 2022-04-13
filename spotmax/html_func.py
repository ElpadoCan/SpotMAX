from functools import wraps

def _tag(tag_info='p style="font-size:10px"'):
    def wrapper(func):
        @wraps(func)
        def inner(text):
            tag = tag_info.split(' ')[0]
            text = f'<{tag_info}>{text}</{tag}>'
            return text
        return inner
    return wrapper

@_tag(tag_info='i')
def italic(text):
    return text

@_tag(tag_info='b')
def bold(text):
    return text

def tag(text, tag_info='p style="font-size:10pt"'):
    tag = tag_info.split(' ')[0]
    text = f'<{tag_info}>{text}</{tag}>'
    return text

def paragraph(text, font_size='13px'):
    return tag(text, tag_info=f'p style="font-size:{font_size}"')

if __name__ == '__main__':
    text = 'ciao'
    print(paragraph(text))
    print(italic(text))
    print(bold(text))
    print(tag(text))
