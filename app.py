from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64, os
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# load model dan scaler
model = load_model('model_lstm_suhu.h5')
scaler = joblib.load('scaler_suhu.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plot_url = None
    method = None

    if request.method == 'POST':
        # jika user upload file
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            df = pd.read_csv(file)

            # cari kolom suhu (ambil kolom terakhir jika tidak tahu namanya)
            suhu_col = df.columns[-1]
            temps = df[suhu_col].dropna().tolist()[-30:]  # ambil 30 data terakhir
            method = "csv"
        else:
            # input manual
            temps = request.form.get('temps')
            temps = [float(t.strip()) for t in temps.split(',')]
            method = "manual"

        # pastikan jumlah data = 30
        if len(temps) == 30:
            scaled_input = scaler.transform(np.array(temps).reshape(-1, 1))
            X_input = scaled_input.reshape((1, 30, 1))
            pred_scaled = model.predict(X_input)
            pred = scaler.inverse_transform(pred_scaled)[0][0]
            prediction = round(pred, 2)

            # buat plot
            plt.figure(figsize=(8, 4))
            plt.plot(range(1, 31), temps, label='Data Input (30 Hari Terakhir)', color='blue', marker='o')
            plt.plot(31, prediction, label='Prediksi Hari ke-31', color='red', marker='x', markersize=10)
            plt.title('Prediksi Suhu Berdasarkan 30 Hari Terakhir')
            plt.xlabel('Hari ke-')
            plt.ylabel('Suhu (°C)')
            plt.legend()
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
        else:
            prediction = "Data kurang. Harus ada 30 nilai suhu."

    return render_template('index.html', prediction=prediction, plot_url=plot_url, method=method)

if __name__ == '__main__':
    app.run(debug=True)
