from email.policy import default
from typing import List


class Argument:
    def __init__(self, argument):
        details = self._get_details(argument)

    def _get_details(self, argument):
        details = {}
        name, *extras = argument.strip().split(":")
        extras = ":".join(extras)
        dtype, *default = extras.split("=")
        dtype = (dtype[0] if dtype else None,)
        default = (default[0] if default else None,)

        details = dict(
            name=name.strip(),
            dtype=dtype.strip(),
            default=default.strip(),
        )

        return details


class Function:
    def __init__(self, function, file_content):
        self.function = function
        self.file_content = file_content

    @property
    def arguments(self):
        arguments = self.file_content.split(f"{self.function}(")
        arguments = arguments[1]
        arguments = arguments.split(")")
        arguments = arguments[0]
        arguments = [Argument(arg) for arg in arguments.split(",")]
        return arguments


class ReferenceGenerator:
    def __init__(self):
        pass

    def _strip(self, line: str) -> str:
        line = line.strip().replace("def ", "")
        line = line.split("(")[0]
        return line

    def _get_functions(self, file_content: str) -> List[str]:
        return [
            self._strip(line)
            for line in str(file_content).split("\n")
            if "def " in line
        ]

    def gen(self, file_content):
        functions = self._get_functions(file_content)

        for function in functions:
            function = Function(function=function, file_content=file_content)
            arguments = function.arguments

        return file_content
