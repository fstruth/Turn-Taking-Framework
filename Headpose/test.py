def finde_index_der_nullen(lst):
    if 0 in lst:
        index_der_letzten_null = len(lst) - 1 - lst[::-1].index(0)
        index_erste_null = lst.index(0) if 0 in lst else None
        return index_erste_null, index_der_letzten_null
    else:
        return None
    
    

# Beispiel
liste_1 = [0, 0, 0, 1, 1]
liste_2 = [0, 1, 0, 1, 1]

index_erste_null, index_der_letzten_null = finde_index_der_letzten_null(liste_2)
if index_der_letzten_null is not None and all(x == 1 for x in liste_2[index_der_letzten_null+1:]) and index_erste_null is not None and all(x == 1 for x in liste_2[index_erste_null+1:]):
    ergebnis = 1
else:
    ergebnis = 0

print(ergebnis)


