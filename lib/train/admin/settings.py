from lib.train.admin.environment import env_settings


class Settings:

    def __init__(self):
        self.set_default()

    def set_default(self):
        self.env = env_settings()
        self.use_gpu = True


