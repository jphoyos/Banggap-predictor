import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import os
import sys

class BandGapEstimator:
    def __init__(self, master):
        self.master = master
        self.master.title("Band Gap Estimator")
        self.master.geometry("1200x800")

        self.file_path = tk.StringVar()
        self.model_var = tk.StringVar()
        self.models = {}
        self.model_name_map = {
            'AdaBoostRegressor': "AdaBoost",
            'BaggingRegressor': "Bagging",
            'ExtraTreesRegressor': "Extra-Trees",
            'GradientBoostingRegressor': "Gradient Boosting",
            'LGBMRegressor': "LightGBM",
            'RandomForestRegressor': "Random Forest",
            'LassoCV': "Lasso CV",
            'ElasticNetCV': "ElasticNet CV",
            'LinearRegression': "Linear Regression",
            'RANSACRegressor': "RANSAC",
            'XGBRegressor': "XGBoost",
            'KNeighborsRegressor': "K-Nearest Neighbors",
            'SGDRegressor': "SGD",
            'ARDRegression': "ARD",
            'BayesianRidge': "Bayesian Ridge",
            'MLPRegressor': "MLP Neural Network"
        }
        self.load_models()

        self.create_widgets()

    def load_models(self):
        """Carga los modelos disponibles desde la carpeta 'models'."""
        # Si está empaquetado con PyInstaller, usar sys._MEIPASS
        if hasattr(sys, '_MEIPASS'):
            model_dir = os.path.join(sys._MEIPASS, "models")
        else:
            model_dir = "models"
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.joblib'):
                model_name = os.path.splitext(model_file)[0]
                model_path = os.path.join(model_dir, model_file)
                self.models[self.model_name_map.get(model_name, model_name)] = joblib.load(model_path)

    def create_widgets(self):
        """Crea y organiza los widgets en la interfaz."""
        # Frame superior para cargar archivo y seleccionar modelo
        top_frame = ttk.Frame(self.master, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="File (CSV/Excel):").pack(side=tk.LEFT)
        ttk.Entry(top_frame, textvariable=self.file_path, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Find", command=self.load_file).pack(side=tk.LEFT)

        ttk.Label(top_frame, text="Model:").pack(side=tk.LEFT, padx=(10, 0))
        model_dropdown = ttk.Combobox(top_frame, textvariable=self.model_var, values=list(self.models.keys()))
        model_dropdown.pack(side=tk.LEFT, padx=5)
        if self.models:
            model_dropdown.set(list(self.models.keys())[0])

        ttk.Button(top_frame, text="Estimate the Band Gap", command=self.estimate_bandgap).pack(side=tk.LEFT, padx=10)

        # Frame principal para la tabla y el gráfico
        main_frame = ttk.PanedWindow(self.master, orient=tk.VERTICAL)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame para la tabla de resultados
        table_frame = ttk.Frame(main_frame, height=200)
        main_frame.add(table_frame, weight=0)

        self.table = ttk.Treeview(table_frame, show='headings', selectmode='browse')
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.table.grid(column=0, row=0, sticky='nsew')
        vsb.grid(column=1, row=0, sticky='ns')
        hsb.grid(column=0, row=1, sticky='ew')
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        self.table.bind('<<TreeviewSelect>>', self.on_table_select)

        # Frame para el gráfico
        self.plot_frame = ttk.Frame(main_frame)
        main_frame.add(self.plot_frame, weight=1)

        # Botón para descargar resultados
        ttk.Button(self.master, text="Download Results", command=self.download_results).pack(pady=10)

    def load_file(self):
        """Carga el archivo seleccionado por el usuario."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls")])
        if file_path:
            self.file_path.set(file_path)

    def estimate_bandgap(self):
        """Realiza la estimación del bandgap utilizando el modelo seleccionado."""
        if not self.file_path.get():
            messagebox.showerror("Error", "Please select a file.")
            return

        try:
            file_extension = os.path.splitext(self.file_path.get())[1].lower()
            if file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path.get())
            elif file_extension == '.csv':
                self.df = pd.read_csv(self.file_path.get())
            else:
                raise ValueError("File format not supported")

            try:
                self.eV = self.df.columns.astype(float)
            except Exception as e:
                messagebox.showerror("Error", "Colmuns must be the range of hv")    
            # Suponemos que las primeras 901 columnas son características
            X = self.df.values
            if X.shape[1] != 901:
                raise ValueError("Models work with 901 features")
            # Predicción con el modelo seleccionado
            model = self.models[self.model_var.get()]
            model.verbose=0
            y_pred = model.predict(X)

            # Guardamos los resultados
            self.results = pd.DataFrame({"ID": range(1, len(y_pred) + 1), "Bandgap Estimated": y_pred})

            # Actualizar la tabla con los resultados
            self.update_table()
            messagebox.showinfo("Success", "Band gap estimated correctly.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while processing the file: {str(e)}")

    def update_table(self):
        """Actualiza la tabla con los resultados de las predicciones."""
        self.table.delete(*self.table.get_children())
        self.table["columns"] = ["ID", "Spectra ", "Bandgap Estimated"]

        for col in self.table["columns"]:
            self.table.heading(col, text=col)
            self.table.column(col, width=150)

        for i, row in self.results.iterrows():
            # Mostrar solo los tres primeros valores de la curva Tauc
            tauc_values = self.df.iloc[i, :3].values
            tauc_str = f"[{tauc_values[0]:.4f}, {tauc_values[1]:.4f}, {tauc_values[2]:.4f}...]"
            self.table.insert("", "end", values=(i + 1, tauc_str, f"{row['Bandgap Estimated']:.4f}"))

    def on_table_select(self, event):
        """Acción al seleccionar una fila de la tabla: mostrar la curva Tauc."""
        selected_item = self.table.selection()[0]
        item = self.table.item(selected_item)
        id = int(float(item['values'][0])) - 1  # Convertimos el ID y obtenemos el índice
        self.plot_results(id)

    def plot_results(self, index):
        """Grafica la curva Tauc correspondiente al índice seleccionado."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        x = self.df.iloc[index, :]
        bandgap_estimate = self.results.loc[index, 'Bandgap Estimated']

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(self.eV, x, label="Spectra")
        ax.axvline(x=bandgap_estimate, color='r', linestyle='--', label=f"Bandgap Estimated: {bandgap_estimate:.2f} eV")
        ax.set_xlabel("Photon energy, hv (eV)")
        ax.set_ylabel("(F(R_∞)*hv)^(1/γ)")
        ax.set_title(f"Spectra curve- ID {index + 1}")
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def download_results(self):
        """Descarga los resultados estimados en un archivo Excel o CSV."""
        if hasattr(self, 'results'):
            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
            if file_path:
                file_extension = os.path.splitext(file_path)[1].lower()
                if file_extension == '.xlsx':
                    self.results.to_excel(file_path, index=False)
                elif file_extension == '.csv':
                    self.results.to_csv(file_path, index=False)
                messagebox.showinfo("Éxito", f"Results saved in {file_path}")
        else:
            messagebox.showerror("Error", "No results for download.")

    def on_closing(self):
        """Función que se ejecuta al cerrar la ventana."""
        if messagebox.askokcancel("Exit", "Are you sure you want to leave?"):
            self.master.quit()  # Salir del bucle principal de Tkinter
            self.master.destroy()  # Destruir la ventana y salir del proceso

if __name__ == "__main__":
    root = tk.Tk()
    app = BandGapEstimator(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Asociamos el cierre correcto
    root.mainloop()
