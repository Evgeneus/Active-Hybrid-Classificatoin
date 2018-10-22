class Heuristic:

    def __init__(self, params):
        self.B = params['B']
        self.B_al = round(self.B * params['B_al_prop'])
        self.B_al_spent = 0
        self.B_cr_spent = 0

    def update_budget_al(self, money_spent):
        self.B_al_spent += money_spent

    @property
    def is_continue_al(self):
        if self.B_al_spent > self.B_al:
            print('AL-Box finished')
            return False
        else:
            return True

