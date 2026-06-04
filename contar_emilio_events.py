import ROOT
from ROOT import TTreeReader, TTreeReaderValue
import numpy as np

def encontrar_rama_evento(archivo, ruta_arbol):
    """Encuentra automáticamente qué rama contiene el número de evento"""
    f = ROOT.TFile(archivo)
    tree = f.Get(ruta_arbol)
    
    posibles_nombres = ["eventNumber", "event_number", "EvtNumber", "EventID", "eventNum", "Event", "event"]
    lista_ramas = [branch.GetName() for branch in tree.GetListOfBranches()]
    
    print("Ramas disponibles en el tree:")
    for rama in lista_ramas:
        print(f"  - {rama}")
    
    for nombre in posibles_nombres:
        if nombre in lista_ramas:
            print(f"\n✓ Usando rama: {nombre}")
            return nombre
    
    print("\n⚠️ No se encontró una rama de evento típica.")
    print("Por favor, revisa la lista de ramas y escribe el nombre correcto:")
    return input("Nombre de la rama de evento: ")

def contar_eventos_unicos(archivo, ruta_arbol, rama_evento=None):
    if rama_evento is None:
        rama_evento = encontrar_rama_evento(archivo, ruta_arbol)
        
    # Activar multithreading implícito si es necesario
    # ROOT.EnableImplicitMT()
    
    df = ROOT.RDataFrame(ruta_arbol, archivo)
    
    # Extraer directamente a un array de NumPy en C++ y obtener valores únicos
    array_eventos = df.AsNumpy([rama_evento])[rama_evento]
    eventos_unicos = np.unique(array_eventos)
    
    print(f"\nResultado (vía RDataFrame):")
    print(f"  - Total de entradas en el tree: {df.Count().GetValue()}")
    print(f"  - Eventos distintos: {len(eventos_unicos)}")
    
    if len(eventos_unicos) > 0:
        print(f"  - Rango: {np.min(eventos_unicos)} - {np.max(eventos_unicos)}")
        
    return set(eventos_unicos)

# Uso
archivo = "/scratch48/emilio.fernandez/Velo/VELOHits.root"
ruta = "VeloMultiTuple_73eaa531/Clusters"

eventos = contar_eventos_unicos(archivo, ruta)