#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 16:04:41 2024

@author: nata
"""

Donors = ['C1=CC=C(N(C)C)C=C1',#D001
          'C1=CC=C(N2CCOCC2)C=C1',#D002
          'N1CCOCC1',#D003
          'N(C)C',#D004
          'C1=C2C=CC=CC2=C(N(C)C)C=C1',#D005
          'C1=C2C=CC=CC2=CC(N(C)C)=C1',#D006
          'C1=CC(C)=CC(C)=C1',#D007
          'C1=C(OC)C=C(OC)C=C1(OC)',#D008
          'C1=CC(OC)=CC(OC)=C1',#D009
          'N1C(C=CC=C2)=C2N(C4=CC=CC=C4)C3=CC=CC=C13',#D010
          'N1C(C=CC=C2)=C2C(C)(C)C3=CC=CC=C13',#D011
          'N1C(C=CC=C2)=C2C(=O)C3=CC=CC=C13',#D012
          'N1C2=C(C3=C1C=CC(Cl)=C3)C=C(Cl)C=C2',#D013
          'N1C(C=CC=C2)=C2SC3=CC=CC=C13',#D014
          'N1C(C=CC=C2)=C2OC3=CC=CC=C13',#D015
          'N(C)C1=CC=CC=C1',#D016
          'C1C2=C(C3=C1C=CC=C3)C=CC=C2',#D017
          'Si1C2=C(C3=C1C=CC=C3)C=CC=C2',#D018
          'P(C2=CC=CC=C2)C1=CC=CC=C1',#D019
          'N1C2=C(C3=C1N=CC=C3)C=CC=C2',#D020
          'N1C2=C(C3=C1C=CN=C3)C=NC=C2',#D021
          'N1C2=C(C3=C1C=CN=C3)C=CC=C2',#D022
          'N1C2=C(C3=C1C(C)=CC=C3)C=CC=C2(C)',#D023
          'C(C=C1)=CC=C1N2C3=C(C=CC=C3)C4=C2C=CC=C4',#D024
          'N(C1=CC=CC=C1)C2=CC=CC=C2',#D025
          'N(C1=CC=C(C)C=C1)C2=CC=C(C)C=C2',#D026
          'N(C1=CC=C(F)C=C1)C2=CC=C(F)C=C2',#D027
          'N1C2=C(C3=C1C=CC=C3)C=CC=C2',#D028
          'N1C2=C(C3=C1C=CC(C)=C3)C=C(C)C=C2',#D029
          'N1C2=C(C3=C1C=CC(F)=C3)C=C(F)C=C2',#D030
          ]

Acceptors = ['C1=CON=C1',#A1_A001_A002_A003
             'C1=COC=N1',#A2_A004_A005_A006
             'C1=CSC=N1',#A6_A007_A008_A009
             'C1=CC(C=CC=C2)=C2C3=C1C=CC=C3',#A010
             'C1=NC=C2C(NC=C2)=C1',#A012
             'C(C=C1)=CC2=C1N=C(O2)C3=CC=CC=C3',#A013
             'C(C=C1)=CC2=C1N=CO2',#A014_A015
             'C1=CC=C2C(SC=N2)=C1',#A016_A017
             'C(C=C12)=CC=C2N=CN1C',#A018_A019
             'C1=C2SC=NC2=CC3=C1N=CS3',#A020_A023
             'C1=C2OC=NC2=CC3=C1N=CO3',#A021_A024
             'C1=C2NC=NC2=CC3=C1NC=N3',#A025
             'C1=C2OC=CC2=CC3=C1C=CO3',#A022_A026
             'C(=CC=C1C2=CC=C3)C=C1SC2=N3',#A027
             'C(=CC=C1C2=CC=C3)C=C1S(=O)(=O)C2=C3',#A028
             'C1=CC=C2C(SC3=C2C=CC(C#N)=C3)=C1',#A029 
             'C(=CC=C1C2=CC=C3)C=C1SiHC2=N3',#A030
             'C(=NC=C1C2=CN=C3)C=C1SiC2=C3',#A031
             'C(=CC=C1C2=CC=C3)C=C1OC2=N3',#A032
             'C1=CC(C#N)=C(C#N)C(C#N)=C1',#A033
             'C(C=C1)=CC=C1C(=O)C(F)(F)(F)',#A034
             'C(C=C1)=CC=C1C=O',#A035
             'C(C=C1)=CC=C1C(=O)C',#A036
             'C(C=C1)=CC=C1S(=O)(=O)C',#A037
             'C1=C(C=O)C=CC(C=O)=C1',#A038
             'C1=C(C#N)C=C(C#N)C=C1(C#N)',#A039
             'C(C=C1)=CC=C1P(C2=CC=CC=C2)(C3=CC=CC=C3)=O',#A040
             'C1=CC=C2C=CC=C(C(=O)C3(=O))C2=C13',#A041
             'C1=C(F)C(F)=C(F)C(F)=C1(F)',#A042
             'C1=C(F)C(F)=CC(F)=C1(F)',#A043
             'C1=CC(C(=O)NC2(=O))=C2C=C1',#A044
             'C1=CC(C(=O)C=CC2(=O))=C2C=C1',#A045_A046
             'C1=CC(=O)C=CC1(=O)',#A047
             'C1=CC(=O)NC1(=O)',#A048_A049
             'C(C=C1)=CC=C1C=C3C(=C(C#N)C#N)C2=CC=CC=C2C3(=O)',#A050
             'C1=CC(C(=O)C=CN2)=C2C=C1',#A051
             'C(C=C1(C))=CC(C)=C1B2C3=CC=CC=C3B(C4=C2C=CC=C4)C5=C(C)C=CC=C5C',#A052
             'B(C2=C(C=C(C=C2C)C)C)C2=C(C=C(C=C2C)C)C',#A053
             'C(C=C12)=CC=C1C(=C(C#N)C#N)C3=CC=CC=C3C2(=O)',#A054
             'C(S1)=CC=C1C=C(C#N)C#N',#A055
             'C(S1)=CC=C1C=C3C(=C(C#N)C#N)C2=CC=CC=C2C3(=O)',#A056
             'C(C=C1)=NC=C1C2=CN=CC=C2',#A057
             'C1=CC2=C3C(C4=NC5=CC(C#N)=C(C#N)C=C5N=C24)=CC=CC3=C1',#A058
             'C(C=C1)=CC(C2=C3C=CC=C2)=C1C4=C3N=C(C#N)C(C#N)=N4',#A059_A060
             'C(N=C1)=NC=C1C#N',#A061
             'C1=CC=CC=N1',#A062_A063             
             'C1=CC(C=CC2=C3N=CC=C2)=C3N=C1',#A064_A065
             ]

Others = ['C1=CC=CC=C1',#O001
          'C1=CC=CS1',#O002
          'C#C',#O003
          'C=C',#O004
          'O',#O005
          'S',#OO006
          'C1=CC=CO1',#O007
          'C1=CC=CN1C',#O008
          'C1=CC2=C(S1)C=CS2',#O009
          'C(S1)=CC=C1C2=CC=CS2',#O010
          'C(=CC=C1C2=CC=C3)C=C1OC2=C3',#O011
          'C(=CC=C1C2=CC=C3)C=C1SC2=C3',#O012
          ]
