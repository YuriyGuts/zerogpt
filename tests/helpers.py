import importlib


class MissingImplementation:
    def __init__(self, import_name: str, reason: BaseException | None = None):
        self.import_name = import_name
        self.reason = reason

    def _fail(self):
        message = f"Missing implementation: {self.import_name}"
        if self.reason is not None:
            message += f"\nOriginal error: {type(self.reason).__name__}: {self.reason}"
        raise AssertionError(message)

    def __call__(self, *args, **kwargs):
        self._fail()

    def __getattr__(self, name):
        return MissingImplementation(f"{self.import_name}.{name}", self.reason)

    def __getitem__(self, key):
        return MissingImplementation(f"{self.import_name}[{key!r}]", self.reason)

    def __iter__(self):
        self._fail()

    def __bool__(self):
        self._fail()

    def __repr__(self):
        return f"<MissingImplementation {self.import_name}>"


def maybe_import(module_name: str, attr_name: str | None = None):
    import_name = module_name if attr_name is None else f"{module_name}.{attr_name}"

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Tolerates:
        # - the target module being missing
        # - any dependency imported by that module being missing
        return MissingImplementation(import_name, exc)

    if attr_name is None:
        return module

    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        return MissingImplementation(import_name, exc)
