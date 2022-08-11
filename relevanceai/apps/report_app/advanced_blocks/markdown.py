import uuid

from typing import Any, Dict, List, Union

from relevanceai.apps.report_app.blocks import ReportBlocks


class MarkdownBlock(ReportBlocks):
    def _format_block(self, obj):
        from marko import block
        from marko.block import BlockElement
        from marko.inline import InlineElement

        block = dict(
            type=self._get_marko_block_type(obj),
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

    def _format_inline(self, obj):
        block = dict(
            type="text",
            marks=[],
            attrs=dict(id=str(uuid.uuid4())),
        )
        from marko import inline

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

    def _get_marko_block_type(self, obj) -> Union[None, str]:
        from marko import block

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

        return None

    def markdown(self, raw: str, add: bool = True) -> List[Dict[str, Any]]:
        try:
            import marko
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "`pip install marko` to use markdown functionality in report app"
            )

        markdown = marko.parse(raw)
        blocks = []
        for child in markdown.children:
            b = dict(
                type="appBlock",
                attrs=dict(id=str(uuid.uuid4())),
                content=[self._format_block(child)],
            )
            blocks.append(b)
            if add:
                self.contents.append(b)

        return blocks
