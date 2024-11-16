
def with_seed(seed):
    """Decorator that sets the random seed for the decorated function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tf.random_seed(seed):
                return func(*args, **kwargs)
        return wrapper
    return decorator