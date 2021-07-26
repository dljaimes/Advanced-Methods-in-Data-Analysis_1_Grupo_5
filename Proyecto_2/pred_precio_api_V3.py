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

os.chdir(r'C:\Users\Leonardo\Desktop\Leo\Maestria Analitica\Intersemestral 2021\Modelos avanzados 1\Proyecto_2')

# =============================================================================
# Función que predice el precio
# =============================================================================


def prec_precio(año, km,estado,marca):
    
    """
    Predice el precio estimado de un vehículo teniendo en cuenta
    los datos del año del modelo del mismo y su kilometraje actual.
    """
    
    # se carga el modelo 
    gbr =  pickle.load(open('modelo_price_auto_gbV2.pkl', 'rb'))
    
    # se depuran los datos de entrada para que puedan ingresar al modelo
    año = int(año)
    
    # Haciendo el proceso de transformación al dato de State
    
    est_1 = ['DC', 'HI', 'VA', 'CT', 'OH', 'VT', 'MI', 'AZ', 'FL',
       'IN', 'MD', 'NV', 'CA']
    est_2 = ['IL', 'NJ', 'KS', 'DE', 'KY', 'GA', 'PA', 'SC', 'NH',
       'CO', 'ND', 'MO', 'MN']
    est_3 = ['TN', 'WI', 'NC', 'MA', 'NY', 'OR', 'ID', 'WA', 'AK',
       'RI', 'IA', 'OK']
    est_4 = ['UT', 'AL', 'ME', 'NE', 'TX', 'LA', 'NM', 'WV', 'AR',
       'MS', 'SD', 'MT', 'WY']
           
    condiciones = [estado in est_1, estado in est_2, estado in est_3, estado in est_4] 
    valores = [1,2,3,4]
    
    est_aju = np.select(condiciones, valores)
    
    # Se hace el ajuste del dato de la marca
    
    marca_1 = ['mitsubishi', 'scion', 'fiat', 'suzuki', 'mercury', 'pontiac']
    marca_2 = ['nissan', 'kia', 'mini', 'hyundai', 'mazda', 'volkswagen']
    marca_3 = ['jaguar', 'subaru', 'toyota', 'dodge', 'honda', 'chrysler','infiniti']
    marca_4 = ['chevrolet', 'lincoln', 'ford', 'acura', 'buick', 'audi']
    marca_5 = ['lexus', 'cadillac', 'bmw', 'volvo', 'jeep']
    marca_6 = ['ram', 'mercedes-benz', 'gmc', 'freightliner']
    marca_7 = ['bentley', 'tesla', 'land', 'porsche']
    
    condiciones_marca = [marca in marca_1, marca in marca_2, marca in marca_3,
                         marca in marca_4, marca in marca_5, marca in marca_6,
                         marca in marca_7] 
    
    valores_marca = [1,2,3,4,5,6,7]
    
    marca_aju = np.select(condiciones_marca, valores_marca)   
    
    # Se cargan los datos al modelo para hacer predicciones
    
    pred_ = gbr.predict([[año,km,est_aju,marca_aju]])[0]
    
    return pred_

if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Por favor ingresar los datos de Año, Kilometraje, siglas del Estado (en USA) y marca del vehículo (minúscula)')
        
    else:

        año = sys.argv[1]
        km = sys.argv[2]
        estado = sys.argv[3]
        marca = sys.argv[4]

        pred_ = predict(año, km,estado,marca)
        
        print(año, km, estado,marca)
        print('Precio estimado del vehículo es: ', pred_)


# =============================================================================
# Desarrollo de la Aplicación
# =============================================================================

app = Flask(__name__)


@app.route('/precio_est', methods=['GET'])
def precio_predict():
    return {
         "precio_estimado": prec_precio(request.args.get('año'),int(request.args.get('km')),
                                        request.args.get('estado'), request.args.get('marca'))
        }, 200


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8888)
