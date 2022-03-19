"""
Create workflow diagrams.
"""


def create_diagram(workflows):
    try:
        import graphviz
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Install graphviz following instructions here: https://graphviz.org/download/"
        )

    dot = graphviz.Digraph(
        engine="dot",
        name="Experimentation Workflow",
        graph_attr={
            "splines": "ortho",
        },
        node_attr={
            "shape": "rect",
        },
        format="png",
    )

    nodes = []
    for i, workflow in enumerate(workflows):
        input_field = workflow["input_field"] + "-field"
        output_field = workflow["output_field"] + "-field"
        dot.node(
            input_field, workflow["input_field"], style="filled", color="lightgreen"
        )  # doctest: +NO_EXE
        dot.node(
            output_field,
            workflow["output_field"],
            style="filled",
            bgcolor="green",
            color="lightblue",
        )
        dot.edge(
            input_field,
            output_field,
            label=workflow["workflow_alias"],
            color="green",
            shape="box",
        )
    return dot
