from MainUI import MainWindow
from .labels_settings import LabelsSettings
from .user_management import User_management


class Controller:
    def __init__(self):
        self.user_management = None
        self.label_settings = LabelsSettings(self)
        self.mainui = None

        # self.model_settings = ModelSettings(self)

    def show_user_management(self):
        if self.mainui is not None:
            self.close_mainui()
        self.user_management = User_management(self)
        self.user_management.show()

    def close_user_management(self):
        if self.user_management is not None:
            self.user_management.close()
            self.user_management = None

    def show_label_settings(self):
        self.label_settings.show()

    def close_label_settings(self):
        self.label_settings.close()

    def show_mainui(self):
        if self.user_management is not None:
            self.close_user_management()
        self.mainui = MainWindow(self)
        self.mainui.show()

    def close_mainui(self):
        if self.mainui is not None:
            self.mainui.close()
            self.mainui = None
    # def show_model_settings(self):
    #     self.model_settings.show()
    #
    # def close_model_settings(self):
    #     self.model_settings.close()
