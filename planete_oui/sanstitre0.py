# Importing the libraries
#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import calendar

def easter_date(year):
    """
    Calcule la date du jour de Pâques d'une année donnée
    Voir https://github.com/dateutil/dateutil/blob/master/dateutil/easter.py
    
    :return: datetime
    """
    a = year // 100
    b = year % 100
    c = (3 * (a + 25)) // 4
    d = (3 * (a + 25)) % 4
    e = (8 * (a + 11)) // 25
    f = (5 * a + b) % 19
    g = (19 * f + c - e) % 30
    h = (f + 11 * g) // 319
    j = (60 * (5 - d) + b) // 4
    k = (60 * (5 - d) + b) % 4
    m = (2 * j - k - g + h) % 7
    n = (g - h + m + 114) // 31
    p = (g - h + m + 114) % 31
    day = p + 1
    month = n
    return datetime(year, month, day)


def is_holiday(the_date):
    """
    Vérifie si la date donnée est un jour férié
    :param the_date: datetime
    :return: bool
    """
    year = the_date.year
    easter = easter_date(year)
    days = [
        datetime(year, 1, 1),  # Premier de l'an
        easter + timedelta(days=1),  # Lundi de Pâques
        datetime(year, 5, 1),  # Fête du Travail
        datetime(year, 5, 8),  # Victoire de 1945
        easter + timedelta(days=39),  # Ascension
        easter + timedelta(days=49),  # Pentecôte
        datetime(year, 7, 14),  # Fête Nationale
        datetime(year, 8, 15),  # Assomption
        datetime(year, 11, 1),  # Toussaint
        datetime(year, 11, 11),  # Armistice 1918
        datetime(year, 12, 25),  # Noël
    ]
    return the_date in days


df = '2019-01-01T10:00:00.0'
do = datetime.strptime(df, '%Y-%m-%dT%H:%M:%S.%f')
d5 = do.replace(minute=0, hour=0, second=0, microsecond=0).isoformat(' ')
d7 = datetime.strptime(d5, "%Y-%m-%d %H:%M:%S")
the_date = datetime(2019, 1, 1)
timediff = the_date - d7
res = is_holiday(d7)
res2 = is_holiday(the_date)



def business_days(date_from, date_to):
    """
    Générateur retournant les jours ouvrés dans la période [date_from:date_to]
    :param date_from: Date de début de la période
    :param date_to: Date de fin de la période
    :return: Générateur
    """
    while date_from <= date_to:
        # Un jour est ouvré s'il n'est ni férié, ni samedi, ni dimanche
        if not is_holiday(date_from) and date_from.isoweekday() not in [6, 7]:
            yield date_from
            date_from += timedelta(days=1)




