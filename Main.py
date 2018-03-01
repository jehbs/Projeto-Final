#Codigo principal
import pandas as pd
import LTSpice_RawRead as LTSpice

if __name__ == "__main__":

    Circuito ='Sallen Key mc + 4bitPRBS [FALHA].raw'
    Circuito ='Nonlinear Rectfier + 4bit PRBS [FALHA] - 300 - 0.2s.raw'
    Circuito ='CTSV mc + 4bitPRBS [FALHA].raw'
    Circuito= 'Biquad Highpass Filter mc + 4bitPRBS [FALHA].raw'
    Conjunto = LTSpice.principal(Circuito)
    print(Conjunto)
