import uproot

def export_column_names(path):
    with uproot.open(path) as f:
        # En DaVinci, la ruta suele ser 'Carpeta/NombreDelArbol'
        # Basándome en tu mensaje, vamos a probar esta ruta:
        tree_path = "MCDecayTreeTuple/MCDecayTree"
        
        try:
            tree = f[tree_path]
            print(f"\n--- Leyendo variables de: {tree_path} ---")
            
            # .keys() sobre un objeto 'tree' te da los nombres de las columnas
            columnas = tree.keys()
            
            with open("lista_variables.txt", "w") as out:
                for col in columnas:
                    out.write(col + "\n")
            
            print(f"¡Listo! Se han encontrado {len(columnas)} variables.")
            print("Las he guardado en el archivo: lista_variables.txt")
            
            # Mostrar las primeras 10 por pantalla para que veas el formato
            print("\nEjemplo de variables encontradas:")
            for c in columnas[:10]:
                print(f" - {c}")

        except Exception as e:
            print(f"Error: No se encontró '{tree_path}'")
            print("Las claves disponibles en el archivo son:")
            print(f.keys())

# Ejecutamos

path = '/lustre/LHCb/alejandro.rodriguez/DecFiles/DVntuple.root'

export_column_names(path)