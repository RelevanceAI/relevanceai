from email.policy import default
from typing import List


class Argument:
    def __init__(self, argument):
        details = self._get_details(argument)
        for key, value in details.items():
            self.__setattr__(key, value)

    def __repr__(self):
        return repr(self.__dict__)

    def _get_details(self, argument):
        details = {}
        argument = argument.replace("=", ":")

        name, *extras = argument.strip().split(":")
        name = name.strip()
        extras = ":".join(extras)
        dtype, *default = extras.split("=")
        dtype = dtype.strip() if dtype else None
        default = default[0].strip() if default else None

        details = dict(
            name=name,
            dtype=dtype,
            default=default,
        )

        return details


class Function:
    def __init__(self, function, file_content):
        self.function = function
        self.file_content = file_content

    @property
    def name(self):
        return self.function

    @property
    def arguments(self):
        arguments = self.file_content.split(f"{self.function}(")
        arguments = arguments[1]
        arguments = arguments.split(")")
        arguments = arguments[0]
        arguments = [Argument(arg) for arg in arguments.split(",") if arg.strip()]
        if arguments:
            return arguments
        else:
            return None


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
            name = function.name
            arguments = function.arguments
            print()

        return file_content
