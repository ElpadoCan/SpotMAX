from functools import wraps

def html_tag(tag_info='p style="font-size:10pt"'):
    def wrapper(func):
        @wraps(func)
        def inner(text):
            tag = tag_info.split(' ')[0]
            text = f'<{tag_info}>{text}</{tag}>'
            return text
        return inner
    return wrapper

@html_tag(tag_info='p style="font-size:10pt"')
def html_paragraph_10pt(text):
    return text

@html_tag(tag_info='p style="font-size:11pt"')
def html_paragraph_11pt(text):
    return text

@html_tag(tag_info='i')
def italic(text):
    return text

@html_tag(tag_info='b')
def bold(text):
    return text

def tag(text, tag_info='p style="font-size:10pt"'):
    tag = tag_info.split(' ')[0]
    text = f'<{tag_info}>{text}</{tag}>'
    return text

if __name__ == '__main__':
    text = 'ciao'
    print(html_paragraph_10pt(text))
    print(italic(text))
    print(bold(text))
    print(tag(text))
