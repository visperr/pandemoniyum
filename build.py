import os
import shutil
import PyInstaller.__main__

def build():
    project_root = os.path.dirname(os.path.abspath(__file__))
    dist_dir = os.path.join(project_root, "dist")
    models_src = os.path.join(project_root, "models_data")
    models_dest = os.path.join(dist_dir, "models_data")
    entry_point = os.path.join(project_root, "src", "main.py")

    print("Cleaning previous builds...")
    for folder in ["build", "dist"]:
        path = os.path.join(project_root, folder)
        if os.path.exists(path):
            shutil.rmtree(path)

    print("Compiling executable with PyInstaller...")
    PyInstaller.__main__.run([
        entry_point,
        "--name=PANdemoniYUM_Tracker",
        "--onedir",                       # Better startup times than --onefile with heavy ML libs
        "--noconsole",                    # Runs headless (hidden) by default
        f"--paths={os.path.join(project_root, 'src')}",
        "--clean",
    ])

    print("Copying inference models to dist folder...")
    output_dir = os.path.join(dist_dir, "PANdemoniYUM_Tracker")
    shutil.copytree(models_src, os.path.join(output_dir, "models_data"))

    print("\nBuild complete. Output located at:")
    print(output_dir)

if __name__ == "__main__":
    build()