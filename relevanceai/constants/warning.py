class Warning:
    MODEL_NOT_SUPPORTED = "Model not directly supported. Will try to infer."

    CENTROID_VECTORS = """Centroid vectors are a list. Assuming they are in the order of the cluster labels.\n
        To specify which vectors mapped to which label, place in the format of\n
        {cluster_label: centroid_vector}."""

    CLUSTER_LABEL_NOT_IN_CENTROIDS = "cluster label not detected in centroid vectors"

    LATEST_VERSION = """We noticed you don't have the latest version!
        We recommend updating to the latest version ({latest_version}) to get all bug fixes and newest features!
        You can do this by running pip install -U relevanceai.
        Changelog: {changelog_url}."""

    NO_DOCUMENT_DETECTED = "No document is detected"

    AUTO_GENERATE_IDS = "We will be auto-generating IDs since no id field is detected"

    COLUMN_DNE = "The specified column {} does not exist."

    UPLOAD_FAILED = "Failed to upload."

    FIELD_MISMATCH = "{} does not match {}."

    NCLUSTERS_GREATER_THAN_NDOCS = "You seem to have more clusters than documents. We recommend reducing the number of clusters."

    MISSING_RELEVANCE_NOTEBOOK = "Displaying using pandas. To get image functionality please install RelevanceAI[notebook]."

    INDEX_STRING = "Integer selection of dataframe is not stable at the moment. Please use a string ID if possible to ensure exact selection."

    MISSING_PACKAGE = "You are missing a package."

    DEFAULT_MODEL = "No model selected, defaulting to sklearn implementation of KMeans with 10 clusters"

    MISSING_ALIAS = "No alias is provided. Auto-generating one for you - `{alias}`."
