class PaddleVersionError(Exception):
    message: str

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
