import cv2
import easyocr
import os
import pandas as pd
from datetime import datetime

class AQIAutoDetector:
    def __init__(self):
        # Using GPU (RTX 3050) for fast scene-text detection
        self.reader = easyocr.Reader(['en'], gpu=True)
        
        # Indian CPCB Breakpoints
        self.bps = {
            "PM25": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400), (250, 500, 401, 500)],
            "PM10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400), (430, 500, 401, 500)],
            "CO":   [(0, 1, 0, 50), (1.1, 2, 51, 100), (2.1, 10, 101, 200), (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 100, 401, 500)]
        }

    def get_sub_index(self, conc, pollutant):
        if pollutant not in self.bps or conc <= 0: return 0
        for (clo, chi, ilo, ihi) in self.bps[pollutant]:
            if clo <= conc <= chi:
                return ((ihi - ilo) / (chi - clo)) * (conc - clo) + ilo
        return 500

    def run(self, folder="input_images"):
        base_path = os.path.abspath(folder)
        all_data = []

        for filename in os.listdir(base_path):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            
            img = cv2.imread(os.path.join(base_path, filename))
            if img is None: continue
            
            # --- AUTO DETECTION STEP ---
            # Search the whole image for digits. Detail=1 gives us coordinates.
            raw_results = self.reader.readtext(img, allowlist='0123456789.', detail=1)
            
            # Filter: We only want results that are pure numbers and are in the center-right
            h, w = img.shape[:2]
            detected_numbers = []
            for (bbox, text, prob) in raw_results:
                if text.replace('.', '', 1).isdigit():
                    # Calculate center Y of the text box
                    center_y = sum([p[1] for p in bbox]) / 4
                    center_x = sum([p[0] for p in bbox]) / 4
                    # Focus only on numbers in the middle horizontal area (avoiding units)
                    if 0.35 * w < center_x < 0.65 * w:
                        detected_numbers.append((center_y, float(text)))

            # Sort detected numbers by their vertical position (Top to Bottom)
            detected_numbers.sort()

            # Map sorted numbers to pollutants based on position
            # Based on your image: AQI is top, then HCHO, TVOC, PM2.5, PM10, CO, CO2
            # We look for the last 4 numbers found
            row = {"Filename": filename, "Timestamp": datetime.now().isoformat()}
            
            # Default values
            row.update({"PM25": 0, "PM10": 0, "CO": 0, "CO2": 0})
            
            if len(detected_numbers) >= 4:
                # We take the last 4 detected numeric values (PM2.5, PM10, CO, CO2)
                target_vals = detected_numbers[-4:]
                row["PM25"] = target_vals[0][1]
                row["PM10"] = target_vals[1][1]
                row["CO"]   = target_vals[2][1]
                row["CO2"]  = target_vals[3][1]

            # Calculate AQI
            si_pm25 = self.get_sub_index(row["PM25"], "PM25")
            si_pm10 = self.get_sub_index(row["PM10"], "PM10")
            si_co   = self.get_sub_index(row["CO"], "CO")
            
            row["Final_AQI"] = max(si_pm25, si_pm10, si_co)
            all_data.append(row)
            print(f" {filename} -> AQI: {int(row['Final_AQI'])}")

        pd.DataFrame(all_data).to_csv("aqi_logs.csv", index=False)
        print("\n--- Pipeline Complete! CSV Updated ---")

if __name__ == "__main__":

    AQIAutoDetector().run()
