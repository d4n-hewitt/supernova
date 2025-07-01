def require_fitted(method):
    def wrapper(self, *args, **kwargs):
        if not self.models:
            raise ValueError("The ensemble has not been fitted yet.")
        return method(self, *args, **kwargs)

    return wrapper
