import functools
import traceback
import warnings


def catch_errors(func):
    """
    Decorate function and avoid vector errors.
    Example:
        class A:
            @catch_vector_errors
            def encode(self):
                return [1, 2, 3]
    """

    @functools.wraps(func)
    def catch_vector(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            # Bulk encode the functions as opposed to encode to accelerate the
            # actual function call
            if hasattr(func, "__name__"):
                if "bulk_encode" in func.__name__:
                    # Rerun with manual encoding
                    try:
                        encode_fn = getattr(
                            args[0], func.__name__.replace("bulk_encode", "encode")
                        )
                        if len(args) > 1 and isinstance(args[1], list):
                            return [encode_fn(x, **kwargs) for x in args[1]]
                        if kwargs:
                            # Take the first input!
                            for v in kwargs.values():
                                if isinstance(v, list):
                                    return [encode_fn(x, **kwargs) for x in v]
                    except:
                        traceback.print_exc()
                        pass
            warnings.warn("Unable to encode. Filling in with dummy vector.")
            traceback.print_exc()
            # get the vector length from the self body
            vector_length = args[0].vector_length
            if isinstance(args[1], str):
                return [1e-7] * vector_length
            elif isinstance(args[1], list):
                # Return the list of vectors
                return [[1e-7] * vector_length] * len(args[1])
            else:
                return [1e-7] * vector_length
        return

    return catch_vector
