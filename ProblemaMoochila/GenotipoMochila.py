import numpy as np

class Genotipo:
    
    def __init__(self, genes, bits, utilidad, restriccion, arrTamaños, arrXmax, arrXmin, arrNdecimales, genotipo=None):
        self.restriccion = restriccion
        self.utilidad = utilidad
        self.genes = genes
        self.bits = bits
        if (genotipo is not None) and (len(genotipo) == int(self.bits)):
            self.genotipo = np.array(genotipo)
        else:
            self.genotipo = np.random.randint(0, 2, int(self.bits))
        self.fenotipo = self.setFenotipo(arrTamaños, arrXmax, arrXmin, arrNdecimales)
        self.Z = self.calcular(utilidad)
        self.R = self.calcular(restriccion)
        
    def getFenotipo(self):
        return self.fenotipo
  
    def setFenotipo(self, arrTamaños, arrXmax, arrXmin, arrNdecimales):
        # Inicializar el array para el fenotipo
        num_pares = len(arrTamaños)
        fenotipo = np.empty(num_pares, dtype=float)  # Usar dtype=float para manejar decimales
        start = 0  # Inicializar la posición de inicio en el genotipo
        for i in range(num_pares):
            end = start + int(arrTamaños[i])  # Calcular la posición de finalización
            binario = ''.join(map(str, self.genotipo[start:end]))  # Convertir el segmento a una cadena binaria
            decimal = arrXmin[i] + int(binario, 2) * (arrXmax[i] - arrXmin[i]) / (2 ** arrTamaños[i] - 1)
            decimal = round(decimal, int(arrNdecimales[i]))
            fenotipo[i] = decimal
            start = end  # Actualizar la posición de inicio para la próxima variable de decisión
        return fenotipo

    def getGenotipo(self):
        return self.genotipo
    
    def setGenotipo(self, genotipo, arrTamaños, arrXmax, arrXmin, arrNdecimales):
        try:            
            tamaño = self.getbits()
            if len(genotipo) != tamaño:
                mensaje = f"Tamaño del genotipo incorrecto, debe ser de {tamaño}"
                raise Exception(mensaje)
            self.genotipo = np.array(genotipo)
            self.fenotipo = self.setFenotipo(arrTamaños, arrXmax, arrXmin, arrNdecimales)
            self.Z = self.calcular(self.getUtilidad())
            self.R = self.calcular(self.getRestriccion())
        except Exception as e:
            print("Error: ", e)
                            
    def getZ(self):
        return self.Z
    
    def getR(self):
        return self.R
    
    def getgenes(self):
        return self.genes
    
    def getbits(self):
        return self.bits
    
    def getRestriccion(self):
        return self.restriccion
    
    def getUtilidad(self):
        return self.utilidad
    
    def calcular(self, funcion):
        fenotipo = self.getFenotipo()
        sum = 0
        for i in range(len(funcion)):
            sum+= fenotipo[i] * funcion[i]
        return sum
    
    def mutar(self, prob):
        #Sacamos una probabilidad por gen y si es < a prob muta
        print("Genotipo antes de proceso de mutación: ")
        print(self.getGenotipo())
        for i in range(int(self.getbits())):
            if np.random.random() < prob:
                print(f"Muta!! BIT: {i}")
                if self.genotipo[i] == 0:
                    self.genotipo[i] = 1
                else:
                    self.genotipo[i] = 0
        print("Genotipo despues de proceso de mutación: ")
        print(self.getGenotipo())

##def __init__(self, genes, bits, utilidad, restriccion, arrTamaños, arrXmax, arrXmin, arrNdecimales, genotipo=None):
#genotipo = Genotipo(2,14,np.array([5, 7]), np.array([10, 15]), [7, 7], [5, 5], [-5, -5], [1, 1])
#print(genotipo.getbits())
#print("Genotipo: ", genotipo.getGenotipo())
#print("Fenotipo: ", genotipo.getFenotipo())
#print("Z: ", genotipo.getZ())    
#print("R: ", genotipo.getR())



