import GenotipoSen as genotipo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#Se tiene un a mochila que solo spoprta 15 unidades de peso y se tienen 4 
# objetos que desean tranformase Se busca que se maximice la utilidad de los
# objetos seleccionados.De cada obejo se tiene una utilidad y un peso
# determinado. de cada objeto se puden llevar 0 o 1 unidad

class Algoritmo:
    
    def __init__(self, tamañoPoblacion, genes, bits, utilidad, restriccion, limite, arrTamaños, Xmax, Xmin, Ndecimales):
        poblInicial = np.array([])
        while len(poblInicial) != tamañoPoblacion:
            individuo = genotipo.Genotipo(genes, bits, utilidad, restriccion, arrTamaños, Xmax, Xmin, Ndecimales)
            if individuo.getR() <= limite:
                poblInicial = np.append(poblInicial, individuo)
        self.limite = limite
        self.poblacion = poblInicial
        self.fitness = self.calcularTotalFitness()
        self.mejorIndividuo = self.calcularMejorIndividuo()
        self.utilidad = utilidad
        self.restriccion = restriccion
        self.bits =  bits
        self.genes = genes

    def getPoblacion(self):
        return self.poblacion
    
    def setPoblacion(self, poblacion):
        self.poblacion = poblacion
        self.fitness = self.calcularTotalFitness()
        self.mejorIndividuo = self.calcularMejorIndividuo()

    def getIndividuo(self, pos):
        return self.poblacion[pos]
        
    def setIndividuo(self, pos, individuo):
        self.poblacion[pos] = individuo
        
    def getTamañoPoblacion(self):
        return len(self.poblacion)
    
    def getMejorIndividuo(self):
        return self.mejorIndividuo
    
    def getFitness(self):
        return self.fitness
    
    def getUtilidad(self):
        return self.utilidad
    
    def getRestriccion(self):
        return self.restriccion
    
    def getgenes(self):
        return self.genes
    
    def getbits(self):
        return self.bits
    
    def calcularTotalFitness(self):
        total = 0
        for i in self.poblacion:
            total += i.getZ()
        return total
    
    def calcularMejorIndividuo(self):
        mejor = self.poblacion[0]
        for i in self.poblacion:
            if i.getZ() > mejor.getZ():
                mejor = i
        return mejor

    def printGenotipoPoblacion(self):
        j = 1
        for i in self.poblacion:
            print("individuo " + str(j) + ": " + str(i.getGenotipo()))
            j = j + 1
            
    def printFenotipoPoblacion(self):
        j = 1
        for i in self.poblacion:
            print("individuo " + str(j) + ": " + str(i.getFenotipo()))
            j = j + 1
    
    def ruleta(self):
        acumulado = self.getProbAcumulada()
        print("Este es el acumulado \n",acumulado)
        escoje = np.random.rand()
        print("escoje:      ", escoje)
        padre = None 
        for i in range(len(acumulado)):
            if escoje < acumulado[i]:
                padre = self.getIndividuo(i)
                break    
        return padre
    
    def getProbAcumulada(self):
        total = self.getFitness()
        acumulado = np.array([])
        sum = 0
        for i in self.getPoblacion():
            prob = (i.getZ()/total)
            sum += prob            
            acumulado = np.append(acumulado, sum)
        return acumulado
    
    def torneo(self):
        z = -1
        padre = None    
        tamaño = self.getTamañoPoblacion()
        for i in range(3):
            seleccionado = np.random.randint(0, tamaño)
            individuo = self.getIndividuo(seleccionado)
            print("Seleccionado individuo ", seleccionado, " :", individuo.getFenotipo())
            if individuo.getZ() > z:
                padre = individuo
        if padre is None:
            padre = self.getIndividuo(-1) # Si no se encuentra un padre, selecciona el último individuo    
        return padre
        
    def seleccion(self, metodo):
        if metodo == 0:
            print("Padres seleccionados por metodo ruleta")
            p1 = self.ruleta()  
            p2 = self.ruleta() 
        else:
            print("Padres seleccionados por metodo torneo")
            p1 = self.torneo()
            p2 = self.torneo()
        return p1,p2
    
    def cruce(self, prob, p1, p2):
        print("\nProbabilidad de cruce de individuos: ", prob)
        print("padre 1:", p1)
        print("padre 2:", p2)
        a1 = np.random.rand()
        if a1 < prob:
            print("Mas grande", prob, "que ", a1, "-> Si Cruzan")
            bit = self.corteGenotipo()  # Se recibe el bit en que se hara el corte
            tamaño = self.getTamañoPoblacion()
            temp1 = p1[0:bit].copy()  # Usar .copy() para evitar modificar el padre original
            temp2 = p1[bit:tamaño].copy()
            print(temp1, temp2)
            temp3 = p2[0:bit].copy()
            temp4 = p2[bit:tamaño].copy()
            print(temp3, temp4)
            hijo1 = np.concatenate((temp1, temp4))
            hijo2 = np.concatenate((temp3, temp2))
        else:
            print("Menor", prob, "que ", a1, "-> NO Cruzan")
            hijo1 = p1.copy()
            hijo2 = p2.copy()
        return hijo1, hijo2
            
    def corteGenotipo(self):
        i = self.getMejorIndividuo()
        tamaño = int(i.getbits())
        n = 1 /(tamaño -1)
        prob = np.random.rand()
        for i in range(1, tamaño):
            if prob < n * i:
                print("\nCorte en bit: " + str(i))
                return i
    def evaluar(self, genotipo):
        #Recibimos un genotipo si este es mayor que el limite lo volvemos None
        if genotipo.getR() > self.limite:
            return None
        return genotipo

    def printInfoPoblacion(self):
        print("{:<17} {:<30} {:<20} {:<10} {:<10}".format("Individuo", "Genotipo", "Fenotipo", "Z", "R"))
        print("-" * 85)
        j = 1
        for individuo in self.poblacion:
            genotipo = individuo.getGenotipo()
            fenotipo = individuo.getFenotipo()
            z = individuo.getZ()
            r = individuo.getR()
            # Convertir las listas a cadenas antes de formatearlas
            genotipo_str = str(genotipo)
            fenotipo_str = str(fenotipo)
            z_str = str(z)
            r_str = str(r)

            print(f"individuo {j}: {genotipo_str:<30} {fenotipo_str:<20} {z_str:<10} {r_str:<10}")
            j += 1

               
    def iterar(self, metodo, Pcruce, Pmutacion, elitismo, iteraciones, arrTamaños, arrXmax, arrXmin, arrNdecimales):
        print("Población inicial")
        self.printGenotipoPoblacion()
        print("\n")
        historial = []
        mejorI = self.getMejorIndividuo()
        avmejorZ = mejorI.getZ()
        historial.append([mejorI.getFenotipo(), mejorI.getZ(), mejorI.getR(), (self.getFitness()/self.getTamañoPoblacion()), avmejorZ])
        ejex = np.array([])
        ejey = np.array([])
        ejez = np.array([])
        
        ejex = np.append(ejex, historial[0][1])
        ejez = np.append(ejez, historial[0][4])
        ejey = np.append(ejey, historial[0][3])
        
        plt.ion()  # Habilita el modo interactivo
        fig, ax = plt.subplots()
        
        ax.set_ylabel('Valor de Z')
        ax.set_xlabel('Iteraciones')
        ax.set_title('Z con el pasar de las iteraciones')
        
        for i in range(iteraciones):
            print("Iteración: ", i+1)
            if elitismo:
                nuevaGeneracion = np.array([mejorI])
            else:
                nuevaGeneracion = np.array([])
            tamañoPoblacion = self.getTamañoPoblacion()
            
            while len(nuevaGeneracion) != tamañoPoblacion:
                print("Seleccion de padres")
                p1, p2 = self.seleccion(metodo)
                print("Padre 1: ", p1.getGenotipo())
                print("Padre 2: ", p2.getGenotipo())
                print("Cruce de padres")
                h1, h2 = self.cruce(Pcruce, p1.getGenotipo(), p2.getGenotipo())
                h1 = genotipo.Genotipo(self.getgenes(), self.getbits(), self.getUtilidad(), self.getRestriccion(), arrTamaños, arrXmax, arrXmin, arrNdecimales, h1)
                h2 = genotipo.Genotipo(self.getgenes(), self.getbits(), self.getUtilidad(), self.getRestriccion(), arrTamaños, arrXmax, arrXmin, arrNdecimales, h2)
                print("Mutación de hijos")
                h1.mutar(Pmutacion)
                h2.mutar(Pmutacion)
                print("Evaluación de hijos")
                h1 = self.evaluar(h1)
                h2 = self.evaluar(h2)
                if h1 is not None and len(nuevaGeneracion) != tamañoPoblacion:
                    print("Hijo 1 factible: ", h1.getGenotipo())
                    nuevaGeneracion = np.append(nuevaGeneracion, h1)
                else:
                    if h1 is None:
                        print("Hijo 1 no factible")
                    else:
                        print("Generacion completada Hijo 1 no agregado")

                if h2 is not None and len(nuevaGeneracion) != tamañoPoblacion:
                    print("Hijo 2 factible: ", h2.getGenotipo())
                    nuevaGeneracion = np.append(nuevaGeneracion, h2)
                else:
                    if h2 is None:
                        print("Hijo 2 no factible")
                    else:
                        print("Generacion completada hijo 2 no agregado")
            print("Fin iteración: ", i+1)
            self.setPoblacion(nuevaGeneracion)
            mejorI = self.getMejorIndividuo()
            avmejorZ += mejorI.getZ()
            historial.append([mejorI.getFenotipo(), mejorI.getZ(), mejorI.getR(), (self.getFitness()/self.getTamañoPoblacion()), avmejorZ/(i+2)])
            ejex = np.append(ejex, historial[i][1])
            ejez = np.append(ejez, historial[i][4])
            ejey = np.append(ejey, historial[i][3])
            
            ax.clear()  # Limpia el gráfico para actualizar los datos
            ax.plot(ejex, label='The best Z (best-so-far)', color='green')
            ax.plot(ejez, label='Average best Z (off-line)', color='blue')
            ax.plot(ejey, label='Average Z (online)', color='red')
            ax.set_ylabel('Valor de Z')
            ax.set_xlabel('Iteraciones')
            ax.set_title('Z con el pasar de las iteraciones')
            ax.legend()
            plt.pause(0.01)  # Pausa durante 0.1 segundos para que se actualice el gráfico
            
            print("Nueva población")
            self.printInfoPoblacion()
            print("\nMejor individuo población")
            i = self.getMejorIndividuo()
            print(f"Genotipo: {i.getGenotipo()}, Fenotipo: {i.getFenotipo()}, Utulidad: {i.getZ()}, Peso: {i.getR()}")
            
        column_names = ["Fenotipo", "Utilidad", "Peso", "on-line", "off-line"]
        df = pd.DataFrame(historial, columns=column_names)
        print("\n",df)
        plt.ioff()
        plt.show()
        
def genTamaños(max, min, n, arrTamaños, arrXmax, arrXmin, arrNdecimales):
    tamaño = math.log2(1 + (max - min) * 10**n) #Calcula el tamaño de cada gen
    tamaño = math.ceil(tamaño) #Redondea el valor hacia arriba
    arrTamaños = np.append(arrTamaños, tamaño)
    arrXmax = np.append(arrXmax, max)
    arrXmin = np.append(arrXmin, min)
    arrNdecimales = np.append(arrNdecimales, n)
    return arrTamaños, arrXmax, arrXmin, arrNdecimales

def genNumTamaños(array):
    sum=0
    for i in range(len(array)):
        sum+= array[i]
    return sum
    
#Z = 4 * x1 + 5 * x2 + 6 * x3 + 3 * x4
#R = 7 * x1 + 6 * x2 + 8 * x3 + 2 * x4

# self, metodo, Pcruce, Pmutacion, elitismo, iteraciones

metodo = 0 # 0 = Ruleta de otro modo torneo
Pcruce = 0.8
Pmutacion = 0.15
elitismo = False
restriccion = np.array([0, 0])
utilidad =    np.array([10, 15])
limite = 1
iteraciones = 100
arrTamaños = np.array([])
arrXmax = np.array([])
arrXmin  = np.array([])
arrNdecimales = np.array([])

arrTamaños, arrXmax, arrXmin, arrNdecimales = genTamaños(2, 0, 2, arrTamaños, arrXmax, arrXmin, arrNdecimales)
arrTamaños, arrXmax, arrXmin, arrNdecimales = genTamaños(5, 2, 2, arrTamaños, arrXmax, arrXmin, arrNdecimales)

genes = len(arrTamaños) # Variables de decisión
tamañoPoblacion = genNumTamaños(arrTamaños) * 2

mochila = Algoritmo(tamañoPoblacion, genes, genNumTamaños(arrTamaños), utilidad, restriccion, limite, arrTamaños, arrXmax, arrXmin, arrNdecimales)
mochila.iterar(metodo, Pcruce, Pmutacion, elitismo, iteraciones, arrTamaños, arrXmax, arrXmin, arrNdecimales)