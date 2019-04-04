class PointSwitchPolicy:

    def __init__(self, B, switch_point):
        self.B = B
        self.B_al = round(self.B * switch_point)
        self.B_crowd = self.B - self.B_al
        self.B_al_spent = 0
        self.B_crowd_spent = 0

    def update_budget_al(self, money_spent):
        self.B_al_spent += money_spent

    def update_budget_crowd(self, money_spent):
        self.B_crowd_spent += money_spent

    @property
    def is_continue_al(self):
        # Hot Fix: To do
        if self.B_al_spent + 300 >= self.B_al:
            return False
        else:
            return True

    @property
    def is_continue_crowd(self):
        if self.B_crowd_spent >= self.B_crowd:
            return False
        else:
            return True
