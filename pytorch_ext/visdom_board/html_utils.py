def add_html_body(html_text_producer):
    def func(*args, **kwargs):
        html_text = html_text_producer(*args, **kwargs)
        return '<!DOCTYPE html> \
                <html> \
                <body> \
                    {} \
                </body> \
                </html>'.format(html_text)
    return func


@add_html_body
def html_progress_bar(bar_name, value, max):
    return '{}: <progress value="{}" max="{}"></progress>'.format(bar_name, value, max)
