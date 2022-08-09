import io
import json
from relevanceai.apps.report_app.blocks import ReportBlocks


class AltairReportBlock(ReportBlocks):
    """
    Adding a Altair block to Report
    """

    def altair(
        self,
        fig,
        title: str = "",
        static: bool = False,
        width: int = 600,
        height: int = 300,
        add: bool = True,
        width_percentage: int = 50,
        **kwargs,
    ):
        try:
            import altair
        except ImportError:
            raise ImportError(
                ".altair requires altair to be installed, install with 'pip install -U altair'."
            )
        if static:
            try:
                import altair_saver

                fp = "_test_.png"
                fig.to_save(fp, format="png")
                self.image(fp, title=title, width_percentage=width_percentage, add=add)
            except ImportError:
                raise ImportError(
                    ".altair 'static=True' requires altair_saver to be installed, install with 'pip install -U altair_saver'."
                )
        else:
            block = {
                "type": "appBlock",
                # "attrs" : {"id": str(uuid.uuid4())},
                "content": [
                    {
                        "attrs": {
                            "height": "auto",
                            "title": title,
                            "width": f"{width_percentage}%",
                            "spec": json.loads(fig.to_json()),
                        },
                        "type": "vegaChart",
                    }
                ],
            }
            if add:
                self.contents.append(block)
            return block

    altair_chart = altair