import uuid

try:
    import marko

    from marko import block
    from marko import inline

    from marko.block import BlockElement
    from marko.inline import InlineElement

    def get_type(obj: BlockElement):
        if isinstance(obj, block.CodeBlock):
            return "paragraph"
        if isinstance(obj, block.BlankLine):
            return "paragraph"
        if isinstance(obj, block.Document):
            return "paragraph"
        if isinstance(obj, block.Heading):
            return "heading"
        if isinstance(obj, block.FencedCode):
            return "test"
        if isinstance(obj, block.List):
            return "orderedList"
        if isinstance(obj, block.ListItem):
            return "listItem"
        if isinstance(obj, block.Quote):
            return "blockquote"
        if isinstance(obj, block.Paragraph):
            return "paragraph"

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`pip install marko` to use markdown functionality in report app"
    )


from typing import Any, Dict, List, Tuple, Union


class ReportBase:
    def __init__(self, name: str, dataset, deployable_id: str = None):
        self.name = name
        self.dataset = dataset
        self.dataset_id = dataset.dataset_id
        self.deployable_id = deployable_id
        self.base_url = "https://cloud.relevance.ai"
        app_config = None
        self.reloaded = False
        if deployable_id:
            try:
                app_config = self.dataset.get_app(deployable_id)
                self.reloaded = True
            except:
                print(
                    f"{deployable_id} does not exist in the dataset, the given id will be used for creating a new app."
                )
            self.deployable_id = deployable_id
        if app_config:
            self.config = app_config["configuration"]
        else:
            self.config = {
                "dataset_name": self.dataset_id,
                "deployable_name": self.name,
                "type": "page",
                "page-content": {"type": "doc", "content": []},
            }
        if self.deployable_id:
            self.config["deployable_id"] = self.deployable_id

    @property
    def contents(self) -> List:
        return self.config["page-content"]["content"]

    def refresh(self, verbose=True, prompt_update: bool = False):
        try:
            app_config = self.dataset.get_app(self.deployable_id)["configuration"]
            self.reloaded = True
        except:
            raise Exception(f"{self.deployable_id} does not exist in the dataset.")
        if self.config == app_config:
            if verbose:
                print("No updates made, no differences detected.")
            return {}
        else:
            if verbose:
                print("Differences deteced app update made, returning the differences")
            self.config = app_config
            return {}

    def reset(self):
        self.config = {
            "dataset_name": self.dataset_id,
            "deployable_name": self.name,
            "type": "page",
            "page-content": {"type": "doc", "content": []},
        }

    def deploy(self, overwrite: bool = False, new: bool = False):
        if new:
            return self.dataset.create_app(
                {k: v for k, v in self.config.items() if k != "deployable_id"}
            )
        else:
            if self.deployable_id:
                status = self.dataset.update_app(
                    self.deployable_id, self.config, overwrite=overwrite
                )
                if status["status"] == "success":
                    return self.dataset.get_app(self.deployable_id)
                else:
                    raise Exception("Failed to update app")
            result = self.dataset.create_app(self.config)
            self.deployable_id = result["deployable_id"]
            self.reloaded = True
            return result

    def app_url(
        self,
    ):
        return f"{self.base_url}/dataset/{self.dataset_id}/deploy/page/{self.dataset.project}/{self.dataset.api_key}/{self.deployable_id}/{self.dataset.region}"

    def gui(self, width: int = 1000, height: int = 800):
        try:
            self.dataset.get_app(self.deployable_id)
        except:
            raise Exception(
                f"{self.deployable_id} does not exist in the dataset, run `.deploy` first to create the app."
            )
        return self._show_ipython(url=self.app_url(), width=width, height=height)

    def _show_ipython(self, url, width: int = 1000, height: int = 800):
        try:
            from IPython.display import IFrame

            return IFrame(url, width=width, height=height)
        except:
            print("This only works within an IPython, Notebook environment.")
            return url

    # def replace_by_id(self, id):
    #     """replace an existing block with another"""
    #     return

    # def generate_code(self):
    #     """generate python code from json"""
    #     return

    def _format_block(self, obj: BlockElement):
        block = dict(
            type=get_type(obj),
            attrs=dict(id=str(uuid.uuid4())),
        )
        if hasattr(obj, "children"):
            block["content"] = []

            for child in obj.children:
                formatted_child = None

                if isinstance(child, BlockElement):
                    formatted_child = self._format_block(child)

                elif isinstance(child, InlineElement):
                    formatted_child = self._format_inline(child)

                block["content"].append(formatted_child)

        if hasattr(obj, "level"):
            block["attrs"]["level"] = obj.level

        if hasattr(obj, "start"):
            block["attrs"]["start"] = obj.start

        return block

    def _format_inline(self, obj: InlineElement):
        block = dict(
            type="text",
            marks=[],
            attrs=dict(id=str(uuid.uuid4())),
        )
        if hasattr(obj, "children"):
            if isinstance(obj.children, list):

                for child in obj.children:
                    if isinstance(child, inline.RawText):
                        block["text"] = child.children

            elif isinstance(obj.children, str):
                block["text"] = obj.children

        if isinstance(obj, inline.StrongEmphasis):
            block["marks"].append(dict(type="bold"))  # type: ignore

        elif isinstance(obj, inline.Emphasis):
            block["marks"].append(dict(type="italic"))  # type: ignore

        return block

    def markdown(self, raw: str) -> None:
        markdown = marko.parse(raw)
        for child in markdown.children:
            b = dict(
                type="appBlock",
                attrs=dict(id=str(uuid.uuid4())),
                content=[self._format_block(child)],
            )
            self.contents.append(b)
