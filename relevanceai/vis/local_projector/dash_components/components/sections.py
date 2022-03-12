from dash import html


def Card(children, **kwargs):
    return html.Section(children, className="card-style")
