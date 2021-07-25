# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 19:46:26 2021

@author: Leonardo
"""


import pandas as pd
import numpy as np
import pickle
import sys
import os
from flask import Flask, request

#os.chdir(r'C:\Users\Leonardo\Desktop\Leo\Maestria Analitica\Intersemestral 2021\Modelos avanzados 1\Proyecto_2')

# =============================================================================
# Función que predice el precio
# =============================================================================


def prec_precio(año, km):
    
    """
    Predice el precio estimado de un vehículo teniendo en cuenta
    los datos del año del modelo del mismo y su kilometraje actual.
    """
    
    # se carga el modelo 
    gbr =  pickle.load(open('modelo_price_auto_gb.pkl', 'rb'))
    
    # se depuran los datos de entrada para que puedan ingresar al modelo
    año = int(año)
    
    # Haciendo el proceso de binning al dato de kilometraje
    
    a = np.array([0, 25855, 42990, 77406, 2457832]) # valores de corte para hacer el binning
        
    condiciones = [(km>=a[0]) & (km<a[1]), (km>=a[1]) & (km<a[2]),
                   (km>=a[2]) & (km<a[3]), (km>=a[3]) & (km<=a[4])] 
    valores = [1,2,3,4]
    
    km_aju = np.select(condiciones, valores, 4)
    
    # Se cargan los datos al modelo para hacer predicciones
    pred_ = gbr.predict([[año,km_aju]])[0]
    
    return pred_

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Por favor ingresar los datos de Año y Kilometraje del vehículo')
        
    else:

        año = sys.argv[1]
        km = sys.argv[2]
        

        pred_ = predict(año, km)
        
        print(año, km)
        print('Precio estimado del vehículo es: ', pred_)


# =============================================================================
# Desarrollo de la Aplicación
# =============================================================================

app = Flask(__name__)


@app.route('/precio_est', methods=['GET'])
def precio_predict():
    return {
         "precio_estimado": prec_precio(request.args.get('año'),request.args.get('km'))
        }, 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
