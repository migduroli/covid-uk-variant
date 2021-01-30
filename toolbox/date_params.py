import numpy as np

OFFSET = 15


def is_leap_year(year):
    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


class Months:
    class Length:
        Jan = 31
        Feb = 29
        Mar = 31
        Apr = 30
        May = 31
        Jun = 30
        Jul = 31
        Aug = 31
        Sep = 30
        Oct = 31
        Nov = 30
        Dec = 31

    class List:
        Numbers: list = np.arange(1, 13).tolist()
        Names: list = ['Jan', 'Feb', 'Mar',
                 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep',
                 'Oct', 'Nov', 'Dec']

        @staticmethod
        def Lengths(year: int) -> list:
            return [31, 28 + int(is_leap_year(year)), 31,
                    30, 31, 30,
                    31, 31, 30,
                    31, 30, 31]