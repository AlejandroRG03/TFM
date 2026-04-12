import uproot

def export_column_names(path, filename="lista_variables.txt"):
    with uproot.open(path) as f:
        # En DaVinci, la ruta suele ser 'Carpeta/NombreDelArbol'
        # Basándome en tu mensaje, vamos a probar esta ruta:
        tree_path = "MCDecayTreeTuple/MCDecayTree"
        
        try:
            tree = f[tree_path]
            print(f"\n--- Leyendo variables de: {tree_path} ---")
            
            # .keys() sobre un objeto 'tree' te da los nombres de las columnas
            columnas = tree.keys()
            
            with open(filename, "w") as out:
                for col in columnas:
                    out.write(col + "\n")
            
            print(f"¡Listo! Se han encontrado {len(columnas)} variables.")
            print(f"Las he guardado en el archivo: {filename}")
            
            # Mostrar las primeras 10 por pantalla para que veas el formato
            print("\nEjemplo de variables encontradas:")
            for c in columnas[:10]:
                print(f" - {c}")

        except Exception as e:
            print(f"Error: No se encontró '{tree_path}'")
            print("Las claves disponibles en el archivo son:")
            print(f.keys())

# Ejecutamos

pathKL0 = '/home3/alejandro.rodriguez/DecFiles/KL0ntuple.root'
pathmup = '/home3/alejandro.rodriguez/DecFiles/mupntuple.root'
pathmun = '/home3/alejandro.rodriguez/DecFiles/mumntuple.root'
pathsignal = '/home3/alejandro.rodriguez/DecFiles/signalntuple.root'
 
export_column_names(pathKL0, "lista_variables_KL0.txt")
export_column_names(pathmup, "lista_variables_mup.txt")
export_column_names(pathmun, "lista_variables_mum.txt")
export_column_names(pathsignal, "lista_variables_signal.txt")