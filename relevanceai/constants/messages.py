class Messages:

    BETA_DOCSTRING = """{}

    .. warning::
        This function is currently in beta and is liable to change in the future. We
        recommend not using this in production systems.

    """
    BETA = "This function currently in beta and may change in the future."

    ADDED_DOCSTRING = """

    .. note::
        This function was introduced in **{}**.

    """

    DEPRECEATED_DOCSTRING = """

    .. note::
        This function has been deprecated as of {}

    """
    DEPRECEATED = "Deprecated. Revert to versions before {} for function. {}"

    FAIL_MESSAGE = """Your API key is invalid. Please login again"""

    WELCOME_MESSAGE = """Welcome to RelevanceAI. Logged in as {}."""

    TOKEN_MESSAGE = "Activation token (you can find it here: {} )\n"

    BUILD_HERE = "Build your clustering app here: "
