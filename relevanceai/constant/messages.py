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
